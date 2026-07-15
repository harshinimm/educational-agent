"""SQLite access layer for the study log.

Single table, per the Phase 0 spec: every study interaction (flashcard
review or photographed paper attempt) is one row. This table is the
dataset every later phase (BKT, DKT, the CNN grader, the RL tutor) trains
and evaluates on, so the schema is intentionally frozen and simple.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "study_log.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    concept TEXT NOT NULL,
    question_id TEXT NOT NULL,
    correct INTEGER NOT NULL,
    time_taken_seconds REAL,
    photo_path TEXT
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def log_interaction(timestamp, concept, question_id, correct, time_taken_seconds, photo_path=None):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO interactions
               (timestamp, concept, question_id, correct, time_taken_seconds, photo_path)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (timestamp, concept, question_id, int(correct), time_taken_seconds, photo_path),
        )
        conn.commit()


def all_interactions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM interactions ORDER BY timestamp").fetchall()
        return [dict(r) for r in rows]


def concept_accuracy():
    """Naive per-concept accuracy: correct / total attempts, no time-decay or ML."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT concept,
                      COUNT(*) AS attempts,
                      SUM(correct) AS correct,
                      AVG(time_taken_seconds) AS avg_time_seconds
               FROM interactions
               GROUP BY concept
               ORDER BY concept"""
        ).fetchall()
        return [dict(r) for r in rows]
