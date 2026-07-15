# Lab Notebook

One running log: every bug, surprise, derivation, and experiment result,
across every phase. By the end this should read like a semester of real
debugging war stories — proof the system was built, not vibecoded.

Newest entries on top. One entry per session/finding; doesn't need to be polished.

---

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
