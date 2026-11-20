"""Regression tests for large-page line suggestion and atomic upload.

Covers the repair for a 1500-box horizontally-staggered OCR page whose line
proposal previously ran cubic time and blew past the 60s worker timeout, and
for the half-written job left behind when the import was interrupted.
"""
import io
import os
import random
import sqlite3
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing import create_app, db  # noqa: E402
from pricing import app as app_module  # noqa: E402
from pricing import layout  # noqa: E402

# The request budget every suggestion must finish inside.
TIME_BUDGET_SECONDS = 60.0


def make_staggered_csv(count, seed=20261120):
    """Worst-case page: every sorted x-span leaves a gap before the next one."""
    random.seed(seed)
    lines = ["框号,文字,x1,y1,x2,y2"]
    for index in range(count):
        x1 = 10 + index * 3 + random.randint(0, 1)
        y1 = 10 + index * 2
        lines.append(f"{index + 1},字{index},{x1},{y1},{x1 + 2},{y1 + 20}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_banded_csv(count, seed=7):
    """Realistic 3-column page with horizontally jittered boxes."""
    random.seed(seed)
    lines = ["框号,文字,x1,y1,x2,y2"]
    for index in range(count):
        band, column = divmod(index, 3)
        y = 10 + band * 40 + random.randint(-2, 2)
        base = (10, 310, 610)[column]
        x1 = base + random.randint(-25, 25)
        lines.append(f"{index + 1},文字{index},{x1},{y},{x1 + 80},{y + 26}")
    return ("\n".join(lines) + "\n").encode("utf-8")


class LargePageSuggestionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_app({
            "TESTING": True,
            "DATABASE": os.path.join(self.tmp.name, "large.sqlite3"),
            "SECRET_KEY": "test",
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def _upload(self, body, name="page.csv"):
        return self.client.post(
            "/upload",
            data={"file": (io.BytesIO(body), name)},
            content_type="multipart/form-data",
            follow_redirects=False,
        )

    def test_staggered_1500_boxes_suggests_inside_budget(self):
        body = make_staggered_csv(1500)
        start = time.perf_counter()
        response = self._upload(body, "stagger1500.csv")
        elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 302)
        self.assertLess(elapsed, TIME_BUDGET_SECONDS,
                        f"upload took {elapsed:.1f}s, must stay within 60s")
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        state = self.client.get(f"/api/jobs/{job_id}").get_json()
        # Every valid box is retained.
        self.assertEqual(len(state["boxes"]), 1500)
        self.assertTrue(state["grid_valid"])
        self.assertEqual(len(state["vertical"]), 4)
        self.assertGreaterEqual(len(state["horizontal"]), 2)

    def test_edit_page_suggest_endpoint_inside_budget(self):
        response = self._upload(make_staggered_csv(1500))
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        start = time.perf_counter()
        result = self.client.post(f"/api/jobs/{job_id}/suggest-lines")
        elapsed = time.perf_counter() - start
        self.assertEqual(result.status_code, 200)
        self.assertLess(elapsed, TIME_BUDGET_SECONDS)
        data = result.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(len(data["vertical"]), 4)
        self.assertGreaterEqual(len(data["horizontal"]), 2)

    def test_suggest_endpoint_reports_503_when_deadline_tripped(self):
        response = self._upload(make_staggered_csv(30))
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        original = app_module._suggest_lines

        def slow_suggest(boxes, deadline=None):
            raise app_module.SuggestTimeout()

        app_module._suggest_lines = slow_suggest
        try:
            result = self.client.post(f"/api/jobs/{job_id}/suggest-lines")
        finally:
            app_module._suggest_lines = original
        self.assertEqual(result.status_code, 503)
        payload = result.get_json()
        self.assertFalse(payload["ok"])
        self.assertIn("超时", payload["error"])
        # Nothing was written: the previously saved layout is untouched.
        before = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertTrue(before["grid_valid"])

    def test_banded_1500_boxes_inside_budget(self):
        start = time.perf_counter()
        response = self._upload(make_banded_csv(1500), "band1500.csv")
        elapsed = time.perf_counter() - start
        self.assertEqual(response.status_code, 302)
        self.assertLess(elapsed, TIME_BUDGET_SECONDS)
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        state = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(len(state["boxes"]), 1500)
        self.assertTrue(state["grid_valid"])

    def test_deadline_guard_raises_timeout(self):
        from pricing.app import SuggestTimeout
        from pricing.csvparse import parse_csv
        boxes = parse_csv(make_staggered_csv(1500)).boxes
        with self.assertRaises(SuggestTimeout):
            app_module._suggest_lines(
                boxes, deadline=time.perf_counter() - 1.0)
        # A healthy deadline returns the full proposal instead.
        horizontal, vertical = app_module._suggest_lines(
            boxes, deadline=time.perf_counter() + 30.0)
        self.assertEqual(len(vertical), 4)
        self.assertGreaterEqual(len(horizontal), 2)


class SuggestionRulePreservationTests(unittest.TestCase):
    """The faster algorithm must keep every existing line-picking rule."""

    def reference_separators(self, boxes, low, high):
        """The original O(n^3) implementation, kept as an oracle."""
        eps = layout.EPS
        spans = sorted((box["x1"], box["x2"], (box["x1"] + box["x2"]) / 2.0)
                       for box in boxes)
        gaps = [index for index in range(len(spans) - 1)
                if spans[index + 1][0] - spans[index][1] > 0]
        best = None
        for first in gaps:
            for second in gaps:
                if second <= first:
                    continue
                groups = [spans[:first + 1],
                          spans[first + 1:second + 1],
                          spans[second + 1:]]
                if any(not group for group in groups):
                    continue
                split_a = (spans[first][1] + spans[first + 1][0]) / 2.0
                split_b = (spans[second][1] + spans[second + 1][0]) / 2.0
                score = 0.0
                for group in groups:
                    centre = sum(entry[2] for entry in group) / len(group)
                    score += sum((entry[2] - centre) ** 2 for entry in group)
                    for start, end, _mid in group:
                        if start < split_a < end or start < split_b < end:
                            score += 1e9
                if best is None or score < best[0]:
                    best = (score, split_a, split_b)
        if best is None:
            step = (high - low) / 3.0
            return [round(low + step, 2), round(low + 2 * step, 2)]
        separators = sorted({
            round(value, 2) for value in (best[1], best[2])
            if low + eps < value < high - eps
        })
        if len(separators) != 2:
            step = (high - low) / 3.0
            return [round(low + step, 2), round(low + 2 * step, 2)]
        return separators

    def test_matches_oracle_on_random_pages(self):
        random.seed(99)
        for _ in range(300):
            count = random.randint(1, 30)
            boxes = []
            for _ in range(count):
                x1 = float(random.randint(0, 100) * 5)
                x2 = x1 + float(random.randint(1, 6) * 5)
                boxes.append({"x1": x1, "x2": x2, "y1": 0.0, "y2": 10.0})
            low = min(box["x1"] for box in boxes)
            high = max(box["x2"] for box in boxes)
            expected = self.reference_separators(boxes, low, high)
            actual = app_module._column_separators(boxes, low, high)
            self.assertEqual(actual, expected)

    def test_tie_keeps_first_gap_pair(self):
        # Two symmetric, equally compact clusters of gap pairs -> identical
        # scores; the first pair in sorted-gap order must win, as before.
        boxes = []
        for x in (0, 10, 100, 110, 200, 210, 300, 310):
            boxes.append({"x1": x, "x2": x + 5, "y1": 0, "y2": 10})
        low = min(b["x1"] for b in boxes)
        high = max(b["x2"] for b in boxes)
        self.assertEqual(
            app_module._column_separators(boxes, low, high),
            self.reference_separators(boxes, low, high),
        )

    def test_equal_thirds_fallback_when_no_three_clusters(self):
        # Overlapping x-spans leave no usable gap pair -> equal thirds.
        boxes = [
            {"x1": 0, "x2": 120, "y1": 0, "y2": 10},
            {"x1": 60, "x2": 180, "y1": 0, "y2": 10},
        ]
        low, high = 0.0, 180.0
        self.assertEqual(
            app_module._column_separators(boxes, low, high),
            [60.0, 120.0],
        )
        self.assertEqual(
            app_module._column_separators(boxes, low, high),
            self.reference_separators(boxes, low, high),
        )

    def test_span_crossing_both_splits_counts_once(self):
        # One wide span crosses every candidate split; the original OR test
        # counts it as one violation per pair, not one per cut.
        boxes = [
            {"x1": 0, "x2": 500, "y1": 0, "y2": 10},
            {"x1": 20, "x2": 30, "y1": 0, "y2": 10},
            {"x1": 120, "x2": 130, "y1": 0, "y2": 10},
            {"x1": 220, "x2": 230, "y1": 0, "y2": 10},
            {"x1": 320, "x2": 330, "y1": 0, "y2": 10},
            {"x1": 420, "x2": 430, "y1": 0, "y2": 10},
        ]
        low, high = 0.0, 500.0
        self.assertEqual(
            app_module._column_separators(boxes, low, high),
            self.reference_separators(boxes, low, high),
        )

    def test_full_suggestion_matches_reference_layout(self):
        body = make_banded_csv(120)
        from pricing.csvparse import parse_csv
        boxes = parse_csv(body).boxes
        horizontal, vertical = app_module._suggest_lines(boxes)
        self.assertEqual(len(vertical), 4)
        self.assertEqual(vertical[0], min(b["x1"] for b in boxes))
        self.assertEqual(vertical[-1], max(b["x2"] for b in boxes))
        # Horizontal separators sit strictly inside the non-overlapping
        # band gaps and are always sorted.
        self.assertEqual(horizontal, sorted(horizontal))
        self.assertLess(horizontal[0], horizontal[-1])


class AtomicUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "atomic.sqlite3")
        self.app = create_app({
            "TESTING": True,
            "DATABASE": self.db_path,
            "SECRET_KEY": "test",
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_job_appears_only_after_lines_are_saved(self):
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(make_banded_csv(30)), "ok.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)
        index = self.client.get("/").data.decode("utf-8")
        self.assertIn("ok.csv", index)
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        state = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(len(state["boxes"]), 30)
        self.assertTrue(state["grid_valid"])

    def test_write_failure_rolls_back_everything_and_reupload_works(self):
        original = db.set_lines

        def failing_set_lines(connection, job_id, horizontal, vertical,
                              commit=True):
            raise db.ImportError("simulated disk failure")

        db.set_lines = failing_set_lines
        try:
            response = self.client.post(
                "/upload",
                data={"file": (io.BytesIO(make_banded_csv(30)), "broken.csv")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
            self.assertIn("导入失败".encode("utf-8"), response.data)
        finally:
            db.set_lines = original

        connection = sqlite3.connect(self.db_path)
        try:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM boxes").fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM lines").fetchone()[0], 0)
        finally:
            connection.close()

        # The task list must not show the failed page.
        self.assertNotIn("broken.csv", self.client.get("/").data.decode("utf-8"))

        # Re-uploading the same page imports cleanly.
        retry = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(make_banded_csv(30)), "broken.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(retry.status_code, 302)
        job_id = int(retry.headers["Location"].rstrip("/").split("/")[-1])
        state = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(len(state["boxes"]), 30)
        self.assertTrue(state["grid_valid"])

    def test_importing_job_is_hidden_and_unreachable(self):
        connection = db.connect(self.db_path)
        try:
            with db.transaction(connection):
                cursor = connection.execute(
                    "INSERT INTO jobs (filename, status)"
                    " VALUES ('ghost.csv', 'importing')")
                ghost_id = cursor.lastrowid
        finally:
            connection.close()
        self.assertNotIn("ghost.csv",
                         self.client.get("/").data.decode("utf-8"))
        self.assertEqual(self.client.get(f"/jobs/{ghost_id}").status_code, 404)
        self.assertEqual(
            self.client.get(f"/api/jobs/{ghost_id}").status_code, 404)


class LegacySchemaMigrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "legacy.sqlite3")
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            );
            CREATE TABLE boxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                row_no INTEGER NOT NULL, box_ref TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                x1 REAL NOT NULL, y1 REAL NOT NULL,
                x2 REAL NOT NULL, y2 REAL NOT NULL,
                UNIQUE (job_id, row_no)
            );
            CREATE TABLE lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                orientation TEXT NOT NULL CHECK (orientation IN ('h','v')),
                position REAL NOT NULL,
                UNIQUE (job_id, orientation, position)
            );
            CREATE TABLE decisions (
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                box_id INTEGER NOT NULL REFERENCES boxes(id) ON DELETE CASCADE,
                kind TEXT NOT NULL CHECK (kind IN ('cell','exclude')),
                target_cell TEXT NOT NULL DEFAULT '',
                overlap_cells TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (job_id, box_id)
            );
            INSERT INTO jobs (filename) VALUES ('legacy.csv');
            INSERT INTO boxes (job_id, row_no, box_ref, text, x1, y1, x2, y2)
                VALUES (1, 1, '1', '糖', 10, 10, 90, 40);
            INSERT INTO lines (job_id, orientation, position)
                VALUES (1,'h',0),(1,'h',100),(1,'v',0),(1,'v',100),
                       (1,'v',200),(1,'v',300);
            """
        )
        connection.commit()
        connection.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_legacy_job_remains_visible_and_valid(self):
        app = create_app({"TESTING": True, "DATABASE": self.db_path,
                          "SECRET_KEY": "test"})
        client = app.test_client()
        self.assertIn("legacy.csv", client.get("/").data.decode("utf-8"))
        state = client.get("/api/jobs/1").get_json()
        self.assertEqual(len(state["boxes"]), 1)
        self.assertTrue(state["grid_valid"])


class InMemoryDatabaseTests(unittest.TestCase):
    def test_full_flow_on_shared_memory_database(self):
        app = create_app({"TESTING": True, "DATABASE": ":memory:",
                          "SECRET_KEY": "test"})
        client = app.test_client()
        body = (
            "框号,文字,x1,y1,x2,y2\n"
            "1,糖,10,10,90,40\n"
            "2,价,210,10,290,40\n"
        ).encode("utf-8")
        response = client.post(
            "/upload",
            data={"file": (io.BytesIO(body), "memory.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 302)
        job_id = int(response.headers["Location"].rstrip("/").split("/")[-1])
        state = client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(len(state["boxes"]), 2)
        self.assertTrue(state["grid_valid"])
        self.assertIn("memory.csv", client.get("/").data.decode("utf-8"))
        self.assertEqual(
            client.post(f"/api/jobs/{job_id}/lines",
                        json={"horizontal": [0, 100],
                              "vertical": [0, 100, 200, 300]}).status_code,
            200,
        )


if __name__ == "__main__":
    unittest.main()
