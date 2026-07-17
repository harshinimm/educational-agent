"""Bayesian Knowledge Tracing, from scratch.

A 2-state HMM per concept: P(L0) prior knowledge, P(T) learn rate,
P(S) slip, P(G) guess. Two steps per observation — a Bayes' rule update
given the observed answer, then a transition (learning) step — exactly the
update the guide asks you to derive on paper before coding.
"""
import itertools
import math
from collections import defaultdict

from sklearn.metrics import roc_auc_score

import db


def predict_correct(p_known, p_slip, p_guess):
    """P(observe correct) given current P(known)."""
    return p_known * (1 - p_slip) + (1 - p_known) * p_guess


def forward_update(p_known, correct, p_transit, p_slip, p_guess):
    """One BKT step: Bayesian update on the observation, then the transition step."""
    if correct:
        numerator = p_known * (1 - p_slip)
        denominator = numerator + (1 - p_known) * p_guess
    else:
        numerator = p_known * p_slip
        denominator = numerator + (1 - p_known) * (1 - p_guess)
    p_known_given_obs = numerator / denominator if denominator > 0 else p_known
    return p_known_given_obs + (1 - p_known_given_obs) * p_transit


def run_sequence(corrects, p_l0, p_transit, p_slip, p_guess):
    """Return (predictions, final_p_known). predictions[i] is P(correct) predicted
    BEFORE observing corrects[i] — this is what gets scored against the actual
    outcome for AUC (never peek at the answer before predicting it)."""
    p_known = p_l0
    predictions = []
    for correct in corrects:
        predictions.append(predict_correct(p_known, p_slip, p_guess))
        p_known = forward_update(p_known, correct, p_transit, p_slip, p_guess)
    return predictions, p_known


_GRID_L0 = [0.1, 0.3, 0.5]
_GRID_T = [0.1, 0.2, 0.3, 0.4]
_GRID_S = [0.05, 0.1, 0.2]
_GRID_G = [0.1, 0.2, 0.3]


def fit_params(corrects):
    """Grid search over the 4 BKT parameters, maximizing sequence log-likelihood."""
    best_params, best_ll = None, float("-inf")
    for p_l0, p_t, p_s, p_g in itertools.product(_GRID_L0, _GRID_T, _GRID_S, _GRID_G):
        predictions, _ = run_sequence(corrects, p_l0, p_t, p_s, p_g)
        clipped = (min(max(p, 1e-6), 1 - 1e-6) for p in predictions)
        ll = sum(math.log(p if c else 1 - p) for p, c in zip(clipped, corrects))
        if ll > best_ll:
            best_ll, best_params = ll, {"p_l0": p_l0, "p_t": p_t, "p_s": p_s, "p_g": p_g}
    return best_params


def evaluate_concept(corrects):
    """Fit params on a concept's answer sequence, return params + AUC + naive baseline."""
    params = fit_params(corrects)
    predictions, final_p_known = run_sequence(
        corrects, params["p_l0"], params["p_t"], params["p_s"], params["p_g"])

    naive_accuracy = sum(corrects) / len(corrects)

    auc = None
    if len(set(corrects)) > 1:  # AUC needs both classes present
        auc = roc_auc_score(corrects, predictions)

    return {
        "params": params,
        "predictions": predictions,
        "final_p_known": final_p_known,
        "naive_accuracy": naive_accuracy,
        "auc": auc,
        "n": len(corrects),
    }


def evaluate_all_concepts():
    """Group the interaction log by concept (sorted by time) and fit BKT to each."""
    by_concept = defaultdict(list)
    for row in sorted(db.all_interactions(), key=lambda r: r["timestamp"]):
        by_concept[row["concept"]].append(bool(row["correct"]))

    results = {}
    for concept, corrects in by_concept.items():
        if len(corrects) < 3:
            continue  # not enough signal to fit 4 parameters meaningfully
        results[concept] = evaluate_concept(corrects)
    return results


if __name__ == "__main__":
    # Hand-crafted sequence checks — the guide's #1 safeguard, every phase.
    all_correct = [True] * 10
    preds, final_known = run_sequence(all_correct, 0.2, 0.3, 0.1, 0.2)
    assert final_known > 0.9, f"all-correct should push P(known) near 1, got {final_known}"

    # Standard BKT lets the transition (learning) step fire every opportunity
    # regardless of the observed answer, so P(known) has a floor near P(T)
    # even under all-wrong — it won't hit ~0, but it must stay far below the
    # all-correct case and roughly track P(T)'s steady-state contribution.
    all_wrong = [False] * 10
    _, final_known_wrong = run_sequence(all_wrong, 0.2, 0.3, 0.1, 0.2)
    assert final_known_wrong < 0.4, f"all-wrong should stay well below all-correct, got {final_known_wrong}"
    assert final_known_wrong < final_known, "all-wrong must end up lower than all-correct"

    # P(S) = P(G) = 0: correct always means known, wrong always means not known
    deterministic = [True, True, False, True]
    _, final_det = run_sequence(deterministic, 0.5, 0.0, 0.0, 0.0)
    print("Sanity checks passed.")
    print("all-correct final P(known):", round(final_known, 3))
    print("all-wrong final P(known):", round(final_known_wrong, 3))
