"""Lightweight adaptive "what to study next" policy.

This is a deliberately simplified stand-in for the guide's full RL tutor
(trained via policy gradient inside a DKT simulator) — that needs a
trustworthy DKT to simulate against, which is out of scope for a 4-day
build. Instead: score each concept from current BKT mastery, forgetting-
curve urgency, and how little it's been practiced, then pick
epsilon-greedily so the policy doesn't tunnel-vision on the same weak spots
forever. It's a real adaptive policy, just not deep RL.
"""
import random

import bkt
import content
import scheduler

WEIGHTS = {"mastery_gap": 0.5, "urgency": 0.35, "exploration": 0.15}


def _concept_scores():
    concepts = content.list_concepts()
    bkt_results = bkt.evaluate_all_concepts()
    queue_by_concept = {row["concept"]: row for row in scheduler.review_queue()}

    scores = {}
    for concept in concepts:
        result = bkt_results.get(concept)
        p_known = result["final_p_known"] if result else 0.0
        n_attempts = result["n"] if result else 0
        queue_row = queue_by_concept.get(concept)
        urgency = (1 - queue_row["predicted_recall"]) if queue_row else 1.0  # untouched = fully urgent
        exploration_bonus = 1 / (1 + n_attempts)
        scores[concept] = (
            WEIGHTS["mastery_gap"] * (1 - p_known)
            + WEIGHTS["urgency"] * urgency
            + WEIGHTS["exploration"] * exploration_bonus
        )
    return scores


def recommend_next(epsilon=0.15):
    """Pick one concept to study next, epsilon-greedily over the priority score."""
    scores = _concept_scores()
    if not scores:
        return None
    if random.random() < epsilon:
        concept = random.choice(list(scores.keys()))
        reason = "exploration pick — keeps the policy from tunnel-visioning on the same weak spots"
    else:
        concept = max(scores, key=scores.get)
        reason = "highest priority: low mastery, overdue for review, or under-practiced"
    return {"concept": concept, "score": scores[concept], "reason": reason, "all_scores": scores}


def weakest_concepts(n=5):
    """Top-n concepts by ascending BKT mastery — feeds the Question Paper
    generation mode's weak-area pre-selection."""
    bkt_results = bkt.evaluate_all_concepts()
    concepts = content.list_concepts()
    ranked = sorted(concepts, key=lambda c: bkt_results[c]["final_p_known"] if c in bkt_results else 0.0)
    return ranked[:n]
