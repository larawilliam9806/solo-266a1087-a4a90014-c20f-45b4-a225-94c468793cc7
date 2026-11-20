"""SQLite storage for uploaded boxes, ruling lines and human decisions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class ImportError_(RuntimeError):
    """A multi-table import failed; the surrounding transaction must roll back."""


# Public alias without shadowing the builtin used elsewhere.
ImportError = ImportError_


# An ``:memory:`` database lives inside one connection, so all requests share
# a single guarded connection in that mode.
_MEMORY_LOCK = threading.Lock()
_MEMORY_CONNECTION = None


def connect(path, timeout=15.0):
    global _MEMORY_CONNECTION
    if path == ":memory:":
        with _MEMORY_LOCK:
            if _MEMORY_CONNECTION is None:
                connection = sqlite3.connect(
                    ":memory:", timeout=timeout, check_same_thread=False)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.isolation_level = None
                connection.executescript(
                    SCHEMA_PATH.read_text(encoding="utf-8"))
                _MEMORY_CONNECTION = connection
            return _MEMORY_CONNECTION

    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # WAL + a busy timeout keep two gunicorn workers from tripping over each
    # other's short writes.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 15000")
    # Transactions are managed explicitly (see ``transaction``); turning the
    # driver's implicit transactions off makes the atomic-import boundary
    # visible and reliable.
    connection.isolation_level = None
    return connection


def init_db(path):
    connection = connect(path)
    try:
        if path != ":memory:":
            connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            _migrate(connection)
        # The shared in-memory connection keeps its schema and must stay open
        # for the lifetime of the process.
    finally:
        if path != ":memory:":
            connection.close()


def _migrate(connection):
    """Bring databases created by older schema versions up to date."""
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
    }
    if columns and "status" not in columns:
        # Existing rows were all fully imported under the old code path.
        connection.execute(
            "ALTER TABLE jobs ADD COLUMN status TEXT NOT NULL DEFAULT 'ready'"
        )


@contextmanager
def transaction(connection):
    """Run one logical unit of work in a single SQLite transaction.

    Commits on success and rolls back on any exception, so an interrupted
    import can never leave a job row without its boxes or lines.
    """
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


@contextmanager
def _unit_of_work(connection, commit):
    """Wrap a standalone write in a transaction unless the caller owns one."""
    if not commit:
        yield
        return
    with transaction(connection):
        yield


def _rows_to_dicts(rows):
    return [dict(row) for row in rows]


def get_job(connection, job_id):
    row = connection.execute(
        "SELECT * FROM jobs WHERE id = ? AND status = 'ready'", (job_id,)
    ).fetchone()
    return dict(row) if row else None


def list_jobs(connection):
    # Importing jobs stay hidden: a page only appears once its boxes AND its
    # proposed ruling lines have been committed together.
    rows = connection.execute(
        "SELECT j.*, (SELECT COUNT(*) FROM boxes b WHERE b.job_id = j.id) AS box_count,"
        " (SELECT COUNT(*) FROM lines l WHERE l.job_id = j.id) AS line_count"
        " FROM jobs j WHERE j.status = 'ready' ORDER BY j.id DESC"
    ).fetchall()
    return _rows_to_dicts(rows)


def create_job(connection, filename, commit=True):
    with _unit_of_work(connection, commit):
        cursor = connection.execute(
            "INSERT INTO jobs (filename, status) VALUES (?, 'importing')",
            (filename,),
        )
        return cursor.lastrowid


def mark_job_ready(connection, job_id, commit=True):
    with _unit_of_work(connection, commit):
        connection.execute(
            "UPDATE jobs SET status = 'ready' WHERE id = ?", (job_id,)
        )


def delete_job(connection, job_id):
    with transaction(connection):
        connection.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def replace_box(connection, job_id, row):
    """Insert one OCR box; UNIQUE(job_id,row_no) makes re-upload idempotent."""
    connection.execute(
        "INSERT OR REPLACE INTO boxes"
        " (job_id, row_no, box_ref, text, x1, y1, x2, y2)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, row["row_no"], row["box_ref"], row["text"],
         row["x1"], row["y1"], row["x2"], row["y2"]),
    )


def add_boxes(connection, job_id, rows, commit=True):
    with _unit_of_work(connection, commit):
        for row in rows:
            replace_box(connection, job_id, row)


def get_boxes(connection, job_id):
    rows = connection.execute(
        "SELECT * FROM boxes WHERE job_id = ? ORDER BY row_no", (job_id,)
    ).fetchall()
    return _rows_to_dicts(rows)


def set_lines(connection, job_id, horizontal, vertical, commit=True):
    with _unit_of_work(connection, commit):
        connection.execute("DELETE FROM lines WHERE job_id = ?", (job_id,))
        for position in horizontal:
            connection.execute(
                "INSERT INTO lines (job_id, orientation, position)"
                " VALUES (?, 'h', ?)",
                (job_id, position),
            )
        for position in vertical:
            connection.execute(
                "INSERT INTO lines (job_id, orientation, position)"
                " VALUES (?, 'v', ?)",
                (job_id, position),
            )


def get_lines(connection, job_id):
    rows = connection.execute(
        "SELECT orientation, position FROM lines WHERE job_id = ?"
        " ORDER BY orientation, position", (job_id,)
    ).fetchall()
    horizontal = [row["position"] for row in rows if row["orientation"] == "h"]
    vertical = [row["position"] for row in rows if row["orientation"] == "v"]
    return horizontal, vertical


def get_decisions(connection, job_id):
    rows = connection.execute(
        "SELECT * FROM decisions WHERE job_id = ?", (job_id,)
    ).fetchall()
    decisions = {}
    for row in rows:
        record = dict(row)
        try:
            record["overlap_cells"] = json.loads(record["overlap_cells"] or "[]")
        except ValueError:
            record["overlap_cells"] = []
        decisions[record["box_id"]] = record
    return decisions


def set_decision(connection, job_id, box_id, kind, target_cell, overlap_cells):
    with transaction(connection):
        connection.execute(
            "INSERT INTO decisions"
            " (job_id, box_id, kind, target_cell, overlap_cells)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (job_id, box_id) DO UPDATE SET"
            " kind = excluded.kind, target_cell = excluded.target_cell,"
            " overlap_cells = excluded.overlap_cells,"
            " created_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (job_id, box_id, kind, target_cell,
             json.dumps(sorted(overlap_cells), ensure_ascii=False)),
        )


def clear_decision(connection, job_id, box_id):
    with transaction(connection):
        connection.execute(
            "DELETE FROM decisions WHERE job_id = ? AND box_id = ?",
            (job_id, box_id),
        )
