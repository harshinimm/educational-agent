"""Forgetting-curve model + review scheduler.

Predicted recall decays exponentially from a concept's current BKT
P(known), with a "stability" (days) that grows with practice — more
correct repetitions means slower forgetting, the same intuition behind
spaced-repetition schedulers. Concepts are flagged for review once
predicted recall drops to ~90%, per the guide's target.
"""
import math
from datetime import datetime, timezone

import bkt
import db

REVIEW_THRESHOLD = 0.9


def _parse_ts(ts_str):
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def predicted_recall(p_known, days_since_review, correct_count):
    stability_days = max(1.0, 3 * p_known * min(correct_count, 10))
    return p_known * math.exp(-days_since_review / stability_days)


def review_queue():
    """Return one row per concept with enough data for BKT to have fit it,
    sorted most-urgent (lowest predicted recall) first."""
    bkt_results = bkt.evaluate_all_concepts()

    by_concept = {}
    for row in db.all_interactions():
        by_concept.setdefault(row["concept"], []).append(row)

    now = datetime.now(timezone.utc)
    queue = []
    for concept, rows in by_concept.items():
        result = bkt_results.get(concept)
        if result is None:
            continue
        p_known = result["final_p_known"]
        last_review = max(_parse_ts(r["timestamp"]) for r in rows)
        correct_count = sum(1 for r in rows if r["correct"])
        days_since = (now - last_review).total_seconds() / 86400
        recall = predicted_recall(p_known, days_since, correct_count)
        queue.append({
            "concept": concept,
            "p_known": round(p_known, 3),
            "days_since_review": round(days_since, 1),
            "predicted_recall": round(recall, 3),
            "due": recall <= REVIEW_THRESHOLD,
        })

    queue.sort(key=lambda r: r["predicted_recall"])
    return queue
