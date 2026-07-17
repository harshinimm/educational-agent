"""Deep Knowledge Tracing — a small LSTM (Piech et al., 2015).

Input at each step: a one-hot encoding of (concept, correctness) — 2N dims
for N concepts. Output: a sigmoid per concept; the value at time t predicts
correctness of whichever concept is asked at t+1 (masked binary
cross-entropy — only the concept actually asked next contributes to the
loss). Trained on a strict chronological split — first ~80% of the log for
training, the trailing slice held out for AUC — never shuffled, per the
guide ("train weeks 1-3, test week 4").
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import db


class DKT(nn.Module):
    def __init__(self, n_concepts, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2 * n_concepts, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_concepts)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h)  # logits, shape (T, n_concepts)


def _build_sequence(rows, concept_to_idx):
    n = len(concept_to_idx)
    x = torch.zeros(len(rows), 2 * n)
    concept_idx = torch.zeros(len(rows), dtype=torch.long)
    correct = torch.zeros(len(rows))
    for i, row in enumerate(rows):
        c_idx = concept_to_idx[row["concept"]]
        concept_idx[i] = c_idx
        correct[i] = 1.0 if row["correct"] else 0.0
        offset = 0 if row["correct"] else n
        x[i, offset + c_idx] = 1.0
    return x, concept_idx, correct


def train_and_evaluate(hidden_size=32, epochs=60, lr=0.01, train_frac=0.8, seed=0):
    torch.manual_seed(seed)
    rows = sorted(db.all_interactions(), key=lambda r: r["timestamp"])
    concepts = sorted({r["concept"] for r in rows})
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    n = len(concepts)

    x, concept_idx, correct = _build_sequence(rows, concept_to_idx)
    total = len(rows)
    train_end = int(total * train_frac)
    if train_end < 5 or total - train_end < 5:
        raise ValueError("Not enough interactions yet for a meaningful train/test split.")

    model = DKT(n, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_batch = x.unsqueeze(0)  # (1, T, 2N)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_batch).squeeze(0)  # (T, N)
        pred_logits = logits[: train_end - 1]
        target_idx = concept_idx[1:train_end]
        target_correct = correct[1:train_end]
        gathered = pred_logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(gathered, target_correct)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_batch).squeeze(0)
        pred_logits = logits[train_end - 1: total - 1]
        target_idx = concept_idx[train_end:total]
        target_correct = correct[train_end:total]
        gathered = pred_logits.gather(1, target_idx.unsqueeze(1)).squeeze(1)
        probs = torch.sigmoid(gathered).numpy()
        actual = target_correct.numpy()

    auc = roc_auc_score(actual, probs) if len(set(actual.tolist())) > 1 else None

    return {
        "auc": auc,
        "naive_accuracy": float(actual.mean()),
        "n_train": train_end,
        "n_test": total - train_end,
    }


if __name__ == "__main__":
    result = train_and_evaluate()
    print(result)
