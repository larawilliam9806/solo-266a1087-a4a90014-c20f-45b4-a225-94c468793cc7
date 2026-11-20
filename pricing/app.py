"""Flask application: old-newspaper price-list entry.

Upload one page of OCR CSV, draw/confirm the ruling lines that split the page
into 品名 / 规格 / 批发价 columns, resolve every box that does not sit wholly
inside a single cell, then download the three-column CSV.
"""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote

from flask import (
    Flask,
    Response,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from . import db
from . import layout
from .csvparse import CsvError, parse_csv

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# A single suggestion must finish comfortably inside gunicorn's 60s worker
# timeout.  The algorithm below is O(n**2); this guard only ever trips on
# pathological inputs and leaves the caller free to fall back / report.
SUGGEST_DEADLINE_SECONDS = 50.0


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-change-me"),
        DATABASE=os.environ.get("DATABASE", "pricing.sqlite3"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    )
    if config:
        app.config.update(config)

    database = app.config["DATABASE"]
    if database != ":memory:":
        Path(database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    db.init_db(database)

    # ------------------------------------------------------------------
    # Per-request connection
    # ------------------------------------------------------------------
    def get_db():
        if "db" not in g:
            g.db = db.connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_db(_exception=None):
        connection = g.pop("db", None)
        # The shared :memory: connection must outlive individual requests.
        if connection is not None and app.config["DATABASE"] != ":memory:":
            connection.close()

    def load_job_or_404(job_id):
        job = db.get_job(get_db(), job_id)
        if job is None:
            abort(404)
        return job

    # ------------------------------------------------------------------
    # State computation shared by pages and API
    # ------------------------------------------------------------------
    def job_state(job_id):
        connection = get_db()
        job = load_job_or_404(job_id)
        boxes = db.get_boxes(connection, job_id)
        horizontal, vertical = db.get_lines(connection, job_id)
        invalid = None
        grid = None
        if len(vertical) == 4 and len(horizontal) >= 2:
            grid = layout.Grid(horizontal, vertical)
        else:
            invalid = _invalid_message(len(horizontal), len(vertical))

        decisions = db.get_decisions(connection, job_id)
        if grid is None:
            evaluated = [
                {
                    "box_id": box["id"], "box_ref": box["box_ref"],
                    "text": box["text"], "x1": box["x1"], "y1": box["y1"],
                    "x2": box["x2"], "y2": box["y2"],
                    "status": "pending", "status_label": "待划格线",
                    "row": None, "col": None, "column": None,
                    "cell_key": "", "candidates": [], "decision": None,
                    "manual": False, "note": "格线无效，暂不归位",
                }
                for box in boxes
            ]
            pending = sum(1 for item in evaluated if (item["text"] or "").strip())
            rows = []
            if not invalid:
                invalid = "尚未确认格线"
        else:
            evaluated = layout.evaluate(grid, boxes, decisions)
            pending = layout.pending_count(evaluated)
            rows = layout.build_rows(evaluated)

        status_counts = {}
        for item in evaluated:
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

        return {
            "job": job,
            "boxes": evaluated,
            "horizontal": horizontal,
            "vertical": vertical,
            "grid_valid": grid is not None,
            "invalid": invalid,
            "pending": pending,
            "rows": rows,
            "status_counts": status_counts,
            "can_export": grid is not None and pending == 0 and bool(boxes),
            "cells": grid.cells() if grid is not None else [],
        }

    def _invalid_message(h_count, v_count):
        parts = []
        if v_count != 4:
            parts.append(f"竖线为 {v_count} 条，需要恰好 4 条")
        if h_count < 2:
            parts.append(f"横线为 {h_count} 条，至少需要 2 条")
        return "格线无效：" + "；".join(parts) if parts else "格线无效"

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    @app.get("/")
    def index():
        jobs = db.list_jobs(get_db())
        return render_template("index.html", jobs=jobs)

    @app.get("/healthz")
    def healthz():
        get_db().execute("SELECT 1")
        return jsonify({"ok": True})

    @app.post("/upload")
    def upload():
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            flash("请选择一个 CSV 文件", "error")
            return redirect(url_for("index"))
        filename = os.path.basename(uploaded.filename)
        raw = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            flash("文件过大（上限 10MB）", "error")
            return redirect(url_for("index"))
        try:
            parsed = parse_csv(raw)
        except CsvError as error:
            for line in [error.message] + error.details:
                flash(line, "error")
            return redirect(url_for("index"))

        # Compute the proposal before opening any write transaction, so a slow
        # suggestion never holds a database lock and never blocks other pages
        # from saving lines or decisions.
        try:
            horizontal, vertical = _suggest_lines(parsed.boxes)
        except SuggestTimeout:
            flash("文字框数量过多，按文字框提议超时，请改用较小页面或手动划设格线",
                  "error")
            return redirect(url_for("index"))

        connection = get_db()
        # One transaction for the whole import: the job row is inserted as
        # "importing" and only becomes visible in the task list after every
        # box and the proposed lines have been written.  Any failure (or a
        # killed worker) rolls the whole thing back, so re-uploading the same
        # page never sees a half-written job.
        try:
            with db.transaction(connection):
                job_id = db.create_job(connection, filename, commit=False)
                db.add_boxes(connection, job_id, parsed.boxes, commit=False)
                db.set_lines(connection, job_id, horizontal, vertical,
                             commit=False)
                db.mark_job_ready(connection, job_id, commit=False)
        except (db.ImportError, sqlite3.Error) as error:
            flash(f"导入失败，未保存任何内容：{error}", "error")
            return redirect(url_for("index"))
        for warning in parsed.warnings[:20]:
            flash(warning, "warning")
        flash(f"已导入 {len(parsed.boxes)} 个文字框，请核对格线", "success")
        return redirect(url_for("job_view", job_id=job_id))

    @app.get("/jobs/<int:job_id>")
    def job_view(job_id):
        job = load_job_or_404(job_id)
        state = job_state(job_id)
        return render_template(
            "job.html", job=job, state=state, state_json=_json_state(state)
        )

    @app.post("/jobs/<int:job_id>/delete")
    def job_delete(job_id):
        load_job_or_404(job_id)
        db.delete_job(get_db(), job_id)
        flash("已删除该任务", "success")
        return redirect(url_for("index"))

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------
    @app.get("/api/jobs/<int:job_id>")
    def api_job(job_id):
        return jsonify(_json_state(job_state(job_id)))

    @app.post("/api/jobs/<int:job_id>/lines")
    def api_set_lines(job_id):
        load_job_or_404(job_id)
        payload = request.get_json(silent=True) or {}
        try:
            horizontal, vertical = layout.normalize_layout(
                payload.get("horizontal", []), payload.get("vertical", [])
            )
        except layout.LayoutError as error:
            # Invalid lines: keep the previous layout and its assignments.
            return jsonify({
                "ok": False,
                "error": error.message,
                "details": error.details,
            }), 400
        db.set_lines(get_db(), job_id, horizontal, vertical)
        return jsonify(_json_state(job_state(job_id)))

    @app.post("/api/jobs/<int:job_id>/suggest-lines")
    def api_suggest_lines(job_id):
        load_job_or_404(job_id)
        boxes = db.get_boxes(get_db(), job_id)
        try:
            horizontal, vertical = _suggest_lines(boxes)
        except SuggestTimeout:
            return jsonify({
                "ok": False,
                "error": "文字框数量过多，按文字框提议超时，请手动划设格线",
                "details": [],
            }), 503
        return jsonify({"ok": True, "horizontal": horizontal, "vertical": vertical})

    @app.post("/api/jobs/<int:job_id>/decisions")
    def api_set_decision(job_id):
        load_job_or_404(job_id)
        payload = request.get_json(silent=True) or {}
        boxes = db.get_boxes(get_db(), job_id)
        box = next((item for item in boxes if item["id"] == payload.get("box_id")), None)
        if box is None:
            return jsonify({"ok": False, "error": "文字框不存在"}), 404
        state = job_state(job_id)
        if not state["grid_valid"]:
            return jsonify({"ok": False, "error": state["invalid"], "details": []}), 409
        grid = layout.Grid(state["horizontal"], state["vertical"])
        item = next(entry for entry in state["boxes"] if entry["box_id"] == box["id"])
        if item["status"] not in ("pending", "stale"):
            return jsonify({
                "ok": False,
                "error": "该文字框已完整落在一格内，无需人工归位",
                "details": [],
            }), 409
        try:
            decision = layout.make_decision(
                grid, item, payload.get("kind"), payload.get("target_cell")
            )
        except layout.LayoutError as error:
            return jsonify({"ok": False, "error": error.message, "details": error.details}), 400
        db.set_decision(get_db(), job_id, box["id"], decision["kind"],
                        decision["target_cell"], decision["overlap_cells"])
        return jsonify(_json_state(job_state(job_id)))

    @app.delete("/api/jobs/<int:job_id>/decisions/<int:box_id>")
    def api_clear_decision(job_id, box_id):
        load_job_or_404(job_id)
        db.clear_decision(get_db(), job_id, box_id)
        return jsonify(_json_state(job_state(job_id)))

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    @app.get("/jobs/<int:job_id>/export.csv")
    def export_csv(job_id):
        state = job_state(job_id)
        if not state["grid_valid"]:
            abort(409, description=state["invalid"])
        if state["pending"] > 0:
            abort(409, description=f"还有 {state['pending']} 个文字框待核对，不能下载")
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(["品名", "规格", "批发价"])
        for row in state["rows"]:
            writer.writerow([row["name"], row["spec"], row["price"]])
        data = "﻿" + buffer.getvalue()
        name = Path(state["job"]["filename"]).stem or "pricelist"
        return Response(
            data,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition":
                    f"attachment; filename*=UTF-8''{_quote_filename(name + '-价目表.csv')}"
            },
        )

    @app.errorhandler(413)
    def too_large(_error):
        flash("文件过大（上限 10MB）", "error")
        return redirect(url_for("index")), 303

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="找不到该任务"), 404

    @app.errorhandler(409)
    def conflict(error):
        return render_template("error.html", code=409,
                               message=getattr(error, "description", "操作被拒绝")), 409

    return app


def _json_state(state):
    job = state["job"]
    return {
        "ok": True,
        "job": {"id": job["id"], "filename": job["filename"]},
        "boxes": state["boxes"],
        "horizontal": state["horizontal"],
        "vertical": state["vertical"],
        "grid_valid": state["grid_valid"],
        "invalid": state["invalid"],
        "pending": state["pending"],
        "rows": state["rows"],
        "status_counts": state["status_counts"],
        "can_export": state["can_export"],
        "cells": state["cells"],
    }


def _quote_filename(name):
    return quote(name)


class SuggestTimeout(Exception):
    """Raised when line suggestion cannot finish inside the request budget."""


def _suggest_lines(boxes, deadline=None):
    """Propose ruling lines from the OCR boxes.

    Vertical: page borders plus two separators chosen so the boxes cluster
    into three columns (splits are only allowed inside empty horizontal gaps).
    Horizontal: page margins plus a separator between adjacent text bands
    that do not overlap vertically.

    The whole computation is purely in-memory and holds no database lock, so
    a large page cannot block other jobs from saving lines or decisions.
    ``deadline`` is an optional perf_counter() wall-clock limit used by the
    callers that must answer inside the 60s worker timeout.
    """
    if not boxes:
        return [], []
    if deadline is None:
        deadline = time.perf_counter() + SUGGEST_DEADLINE_SECONDS

    x1 = min(box["x1"] for box in boxes)
    x2 = max(box["x2"] for box in boxes)
    y1 = min(box["y1"] for box in boxes)
    y2 = max(box["y2"] for box in boxes)

    separators = _column_separators(boxes, x1, x2, deadline)
    vertical = sorted({round(x1, 2), round(x2, 2), *separators})
    if len(vertical) != 4 or x2 - x1 <= layout.EPS:
        # Degenerate page (a single point, or no usable x-spread): propose a
        # nominal 1000-unit-wide page instead of duplicate lines.
        left = x1
        step = 1000.0 / 3.0
        vertical = [round(left, 2), round(left + step, 2),
                    round(left + 2 * step, 2), round(left + 1000.0, 2)]

    # Horizontal: group boxes into visual bands first.
    ordered = sorted(boxes, key=lambda box: (box["y1"], box["x1"]))
    bands = []
    for position, box in enumerate(ordered):
        if (position & 255) == 0 and time.perf_counter() > deadline:
            raise SuggestTimeout()
        placed = False
        for band in bands:
            if layout._same_visual_line(
                {"x1": box["x1"], "y1": box["y1"], "x2": box["x2"], "y2": box["y2"]},
                {"x1": band[0], "y1": band[1], "x2": band[2], "y2": band[3]},
            ):
                band[0] = min(band[0], box["x1"])
                band[1] = min(band[1], box["y1"])
                band[2] = max(band[2], box["x2"])
                band[3] = max(band[3], box["y2"])
                placed = True
                break
        if not placed:
            bands.append([box["x1"], box["y1"], box["x2"], box["y2"]])

    bands.sort(key=lambda band: band[1])
    horizontal = [round(y1, 2), round(y2, 2)]
    for previous, current in zip(bands, bands[1:]):
        if current[1] > previous[3]:
            horizontal.append(round((previous[3] + current[1]) / 2.0, 2))
    # Merge duplicate/near-duplicate lines.
    merged = []
    for value in sorted(horizontal):
        if not merged or abs(value - merged[-1]) > layout.EPS:
            merged.append(value)
    if len(merged) < 2:
        # A single point (or no vertical spread) still needs a page height.
        merged = [round(y1, 2), round(y1 + 1000.0, 2)]
    return merged, vertical


def _column_separators(boxes, low, high, deadline=None):
    """Two x separators that split the boxes into three column clusters.

    Splits are restricted to gaps between the boxes' x-spans, so no box is
    ever cut by a suggested separator.  The pair that minimises within-column
    spread (with a heavy penalty for crossing a box) wins; if the boxes do not
    form three clusters, equal thirds of the page width are used.

    The original implementation rescanned every box for each candidate pair
    (O(n**3)); a 1500-box page could not finish inside the request timeout.
    Here the within-group variance comes from centre prefix sums and the
    crossing penalties are precomputed once per gap, giving O(n**2) while
    producing exactly the same scores, tie handling and equal-thirds
    fallback as before.
    """
    if deadline is None:
        deadline = time.perf_counter() + SUGGEST_DEADLINE_SECONDS

    def equal_thirds():
        step = (high - low) / 3.0
        return [round(low + step, 2), round(low + 2 * step, 2)]

    spans = sorted((box["x1"], box["x2"], (box["x1"] + box["x2"]) / 2.0)
                   for box in boxes)
    n = len(spans)
    gap_after = [index for index in range(n - 1)
                 if spans[index + 1][0] - spans[index][1] > 0]
    # Two cuts need two distinct gaps, and every resulting group must be
    # non-empty: fewer than two gaps cannot form three clusters.
    if len(gap_after) < 2:
        return equal_thirds()

    # Prefix sums of cluster centres make the within-group sum of squared
    # deviations O(1): sum((c-mean)^2) = sum(c^2) - sum(c)^2 / count.
    centres = [entry[2] for entry in spans]
    prefix = [0.0] * (n + 1)
    prefix_sq = [0.0] * (n + 1)
    for index, centre in enumerate(centres):
        prefix[index + 1] = prefix[index] + centre
        prefix_sq[index + 1] = prefix_sq[index] + centre * centre

    def variance(start, end):
        count = end - start + 1
        total = prefix[end + 1] - prefix[start]
        total_sq = prefix_sq[end + 1] - prefix_sq[start]
        return total_sq - total * total / count

    # One candidate cut per gap, together with the indices of boxes whose
    # span the cut line crosses.  Crossing sets are computed up front instead
    # of inside the pair loop, which removes the cubic factor.  The pair
    # penalty counts *distinct* spans crossing either cut, matching the
    # original ``start < split_a < end or start < split_b < end`` test
    # (a span crossing both cuts is one violation, not two).
    cuts = []
    for index in gap_after:
        split = (spans[index][1] + spans[index + 1][0]) / 2.0
        crossed = {
            span_index
            for span_index, (start, end, _mid) in enumerate(spans)
            if start < split < end
        }
        cuts.append((index, split, crossed))

    best = None
    for first_index, (first, split_a, cross_a) in enumerate(cuts):
        if (first_index & 63) == 0 and time.perf_counter() > deadline:
            raise SuggestTimeout()
        for second, split_b, cross_b in cuts[first_index + 1:]:
            score = (
                variance(0, first)
                + variance(first + 1, second)
                + variance(second + 1, n - 1)
                + 1e9 * len(cross_a | cross_b)
            )
            # Strict "<" preserves the original tie behaviour: with equal
            # scores the first pair in sorted-gap order is kept.
            if best is None or score < best[0]:
                best = (score, split_a, split_b)

    separators = sorted({
        round(value, 2) for value in (best[1], best[2])
        if low + layout.EPS < value < high - layout.EPS
    })
    if len(separators) != 2:
        return equal_thirds()
    return separators

