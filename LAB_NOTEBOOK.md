# Lab Notebook

One running log: every bug, surprise, derivation, and experiment result,
across every phase. By the end this should read like a semester of real
debugging war stories — proof the system was built, not vibecoded.

Newest entries on top. One entry per session/finding; doesn't need to be polished.

---

## 2026-07-17 — Pivot to a 4-day hackathon build

Plans changed: this is now a hackathon submission due 2026-07-21 (deployed
demo + public repo + video), not a multi-semester personal project. Cut
the CNN handwriting grader entirely — it would only have been a
disconnected MNIST demo without a real labeled dataset of handwriting to
train a grader on, not worth the time against a 4-day clock. Also
descoped the RL tutor from "policy gradient trained inside a DKT
simulator" down to a lightweight mastery/urgency/exploration-weighted
policy (`tutor_policy.py`) — a real adaptive policy, honestly documented
as a simplified stand-in rather than passed off as the full thing.

Pragmatic engineering-quality tradeoffs made for the time budget (vs. the
original from-scratch learning exercise):
- BKT (`bkt.py`) stayed fully hand-rolled NumPy — cheap, fast, no reason
  to compromise here.
- DKT (`dkt.py`) uses PyTorch rather than hand-written backprop. Writing
  a raw-NumPy LSTM+backward pass wasn't worth 4 days of hackathon time;
  the architecture and training loop are still ours.
- Retrieval (`retriever.py`) uses `sklearn.TfidfVectorizer` for
  vectorization + our own cosine-similarity ranking/concept-filtering.
  Not a vector DB, not an off-the-shelf RAG library — still satisfies the
  "hand-rolled retriever" spirit even without hand-rolling TF-IDF math.

Interesting bug/tuning note: the first synthetic data generator had every
concept converge to BKT P(known) ≈ 1.0 by the end of the log regardless of
difficulty tier, because with ~15-22 opportunities even a "hard" tier
learn rate (P(T)=0.12) gives a >85% cumulative chance of transitioning to
"known" at least once. That flattened the whole "weak areas" story the
Review Queue and Question Paper features depend on. Fix: lowered hard-tier
P(T) to 0.07 and shortened hard-tier attempt counts to 8-14 — leaves 5 of
25 concepts genuinely under-mastered by the end, which is what makes the
tutor policy's weak-area picks (Electrolysis, Redox Reactions, etc. — the
classically hard O-Level Chemistry topics, appropriately) mean something.

DKT vs BKT on the seeded data: DKT held-out AUC 0.843 vs BKT mean AUC
0.756 — DKT ahead, though the two use slightly different evaluation
protocols (BKT: mean of each concept's own causal in-sequence AUC; DKT:
one held-out chronological tail across all concepts jointly), noted
explicitly in the Model Comparison page rather than presented as identical.

## 2026-07-15 — Phase 0 scaffolded

Built the logging skeleton: Streamlit app (`app.py`), SQLite interaction
log (`db.py`), CSV-backed concept/flashcard store (`content.py`).

Decisions:
- Streamlit over Flask/CLI — fastest path to a usable daily-driver UI,
  and `st.camera_input` gives photo capture for free.
- SQLite over CSV for the log — safer for repeated small writes than a
  hand-appended CSV, still trivial to inspect/back up.
- Concepts and flashcards are plain CSV, not in the DB — they're
  hand-edited content, not the interaction log; easy to diff/back up
  separately from the accruing dataset.

Open questions for the checkpoint (after ≥1 week of real use):
- What's annoying about the UX?
- What does naive per-concept accuracy look like — is it a sane
  baseline for BKT (Phase 1) to beat?
