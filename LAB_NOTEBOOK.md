# Lab Notebook

One running log: every bug, surprise, derivation, and experiment result,
across every phase. By the end this should read like a semester of real
debugging war stories — proof the system was built, not vibecoded.

Newest entries on top. One entry per session/finding; doesn't need to be polished.

---

## 2026-07-17 — Switched the LLM layer from Claude to OpenAI

Rewrote `generate.py` to call the OpenAI API instead of Anthropic — the
hackathon expects OpenAI usage specifically (a sponsored-credits track;
the credits didn't come through, so still blocked on getting an actual
key). Kept every function signature identical (`draft_flashcards`,
`draft_flashcards_from_corpus`, `draft_flowchart`, `draft_question_paper`)
so `app.py` needed zero changes beyond the sidebar label. Structured
output via `response_format: {type: json_schema, json_schema: {strict:
true, schema: ...}}` on `chat.completions.create` — same JSON schemas as
before, just a different wire format to request them in. Added `pypdf`
for PDF text extraction on upload (OpenAI's chat completions endpoint
doesn't take raw PDF bytes the way Anthropic's `document` content block
did, so PDFs get extracted to text client-side instead of sent as base64).

Still unverified end-to-end: no OpenAI key to actually test a live call
yet. Everything checked so far is code-level (imports clean, all pages
pass headless AppTest verification) — the real risk (does the structured
output actually come back well-formed, does PDF extraction work on a real
scanned paper) is untested until a key exists.

## 2026-07-17 — Added a real RL tutor (rl_tutor.py)

Reconsidered the earlier scope cut: `tutor_policy.py` is a heuristic, not
RL (no value function, nothing learned from reward — just a formula
recomputed fresh each call). Decided it was worth building genuine RL
given the time was there, and found a shortcut that reuses code already
built: each concept's *fitted* BKT parameters (P(L0), P(T), P(S), P(G))
are already a generative model of how a student answers and learns that
concept — so instead of standing up a separate DKT-based simulator, BKT
itself is the simulator. Real interaction data (~350 rows) isn't enough to
train an RL agent directly; simulated rollouts are.

Environment: state = P(known) vector across concepts, action = which
concept to study next, reward = mastery gain from that one step (`new
P(known) - old P(known)`, via `bkt.forward_update`, the same function the
Knowledge Tracing page uses). Trained a small DQN (2-layer MLP, experience
replay, epsilon-greedy) — 300 simulated episodes trains in a few seconds
on CPU.

Result on the seeded data: RL agent 0.364 average final mastery vs random
0.341 vs greedy-weakest-first 0.329 — greedy-weakest actually loses to
random. Makes sense once you think about it: concepts differ in learn
rate, so the concept with the lowest P(known) isn't always the one with
the best expected return per study step — a "hard" concept with P(T)=0.07
can look most urgent while paying back the least per unit of study time.
Greedy-weakest has no way to see that; the learned Q-function does, at
least partially. Good, honest, explainable RL result — not a huge margin,
but a real and directionally-correct one, reported as such rather than
oversold.

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
