# Study Agent — Phase 0: The Logging Skeleton

A minimal study loop with no ML in it. The point of this phase is to
generate real interaction history — flashcard reviews and photographed
handwritten paper attempts — that later phases (Bayesian Knowledge
Tracing, a hand-rolled LSTM, a CNN handwriting grader, an RL tutor) will
train and evaluate on. Ugly is fine; functional is the bar.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## What's in here

- `app.py` — the Streamlit UI (Study, Cold-Start Diagnostic, Manage
  Concepts, Manage Flashcards, Stats).
- `db.py` — SQLite schema and access for the interaction log
  (`data/study_log.db`, table `interactions`: timestamp, concept,
  question_id, correct, time_taken_seconds, photo_path). This schema is
  the dataset every later phase depends on — don't change it casually.
- `content.py` — CSV-backed concept list and flashcard bank
  (`data/concepts.csv`, `data/flashcards.csv`), both hand-edited via the
  app's Manage pages.
- `data/photos/` — photographed handwritten attempts, referenced by
  `photo_path` in the log.

All of `data/` is gitignored (it's personal study content, auto-created on
first run) except `data/photos/.gitkeep`, which keeps the folder in git.

## Using it

1. **Manage Concepts**: list the topics in your syllabus (30–80, flat list).
2. **Manage Flashcards**: add manual flashcards per concept (LLM-generated
   cards come in a much later phase, deliberately).
3. **Cold-Start Diagnostic**: run once, a single pass across every
   concept, so the first models (Phase 1+) aren't starting from nothing.
4. **Study** daily: flashcard mode for digital review, or "Paper / PYP
   attempt" mode when you work a past-paper question on actual paper —
   snap the working with the camera, mark it, done in about ten extra
   seconds per question.
5. **Stats**: naive per-concept accuracy — the bar every later model
   (BKT, then DKT) has to beat.

## Checkpoint (from the guide)

Study with it for ≥1 week. The log should have a few hundred rows and a
few dozen photos before moving to Phase 1 (BKT). Record in
[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) what's annoying about the UX and what
naive per-concept accuracy looks like.

## Next

Phase 1 — Bayesian Knowledge Tracing, from scratch, on the log this phase
produces.
