# Study Agent — an O-Level Chemistry learning companion

A study app that models what you actually know, tells you what to review
and when, and generates flashcards, flowcharts, and mock exam papers
grounded in your own notes and targeted at your weak spots — not a
flashcard app with an LLM bolted on, but a small pipeline of from-scratch
ML components feeding into a thin LLM layer at the end.

**Live demo:** _add your Streamlit Community Cloud URL here_
**Demo video:** _add your video link here_

## Why this exists

Most "AI study tools" are a prompt wrapped around a chat model. This
project instead builds the actual components a real adaptive tutor needs,
from scratch, and only reaches for an LLM at the very last, thin layer —
once there's something real to ground it in:

```
You study (flashcards, past-paper attempts)
        │
        ▼
[Interaction log]  SQLite: timestamp, concept, correct, time taken
        │
        ├──► [Bayesian Knowledge Tracing]  per-concept P(known), from scratch
        │            │
        │            ├──► [Forgetting-curve scheduler]  review when predicted recall ≈ 90%
        │            │
        │            └──► [Adaptive heuristic policy]  mastery + urgency + exploration → what to study next
        │
        ├──► [Deep Knowledge Tracing]  small LSTM, benchmarked against BKT
        │
        ├──► [RL Tutor]  DQN trained inside a BKT-parameterized simulator
        │
        ▼
[Retriever]  hand-rolled TF-IDF + cosine similarity over your notes, concept-filtered
        │
        ▼
[Claude — generation layer, last, thin]  flashcards / flowcharts / question papers,
 grounded in retrieved notes, targeted at concepts the tutor/RL policy flags as weak
```

## What's implemented

| Component | What it is | File |
|---|---|---|
| Interaction log | SQLite table logging every study attempt | `db.py` |
| Study loop | Flashcard review + photographed paper-attempt logging | `app.py` |
| **Knowledge tracing (BKT)** | 2-state HMM per concept, hand-rolled forward update + grid-search parameter fit, evaluated by AUC | `bkt.py` |
| **Forgetting-curve scheduler** | Exponential recall decay from BKT mastery; flags concepts due for review | `scheduler.py` |
| **Retrieval** | TF-IDF + cosine similarity over a per-concept notes corpus — no vector DB, no RAG library | `retriever.py` |
| **Deep Knowledge Tracing** | Small PyTorch LSTM, masked BCE loss, evaluated on a strict chronological held-out split | `dkt.py` |
| **Adaptive heuristic policy** | Mastery/urgency/exploration-weighted "what to study next" — fast, explainable, not learned | `tutor_policy.py` |
| **RL Tutor** | A real DQN (experience replay, epsilon-greedy), trained inside a simulator built from each concept's own fitted BKT parameters — real interaction data alone isn't enough to train an RL agent, so BKT's P(L0)/P(T)/P(S)/P(G) act as the "model of your own learning" the agent trains against. Beats both a random and a naive greedy-weakest-first baseline in simulated study sessions, because concepts differ in learn rate — greedy-weakest can get stuck grinding on a low-P(T) concept for poor return | `rl_tutor.py` |
| **Generation (LLM, last)** | Claude drafts flashcards, process flowcharts, and mock question papers — all grounded in retrieved notes, all reviewed before saving | `generate.py` |

Cut from scope for the 4-day build: a CNN handwriting grader — would only
have been a disconnected MNIST demo without a real labeled dataset of the
user's handwriting to grade against.

## Try it

The app ships with a seeded synthetic O-Level Chemistry dataset — 25
syllabus concepts, ~5 weeks of simulated study history with deliberately
varied mastery per concept, and a small notes corpus — so every page has
real signal from the moment you open it.

1. **Knowledge Tracing** — see per-concept mastery and the BKT baseline AUC.
2. **Review Queue** — concepts flagged for review as predicted recall decays.
3. **Model Comparison** — train the DKT LSTM live and compare it to BKT.
4. **What to Study Next** — the adaptive heuristic's current top pick.
5. **RL Tutor** — train the DQN live, see it beat random and greedy-weakest
   baselines in simulated sessions, and get its live recommendation.
6. **Generate** — the headline feature: pick "Question Paper" mode and
   watch the weakest concepts (from BKT) get pre-selected, grounded in
   retrieved notes, and turned into a gradeable mock exam.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python seed_data.py             # populate the demo dataset (safe to re-run)
streamlit run app.py
```

The Generate page needs an Anthropic API key — paste one into the sidebar,
or set `ANTHROPIC_API_KEY` as an environment variable.

## Deployment

Deployed on Streamlit Community Cloud, connected to this repo's `main`
branch. `ANTHROPIC_API_KEY` is set as a Cloud secret so the Generate page
works with zero setup for anyone trying the live demo; the max-cards /
max-questions sliders are capped low across all three Generate modes to
bound cost on a public, unauthenticated deploy.

## Origins

This started as a personal, multi-semester "build every ML component from
scratch" project — see [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md) for the running
build log, including the pivot to a 4-day hackathon scope.
