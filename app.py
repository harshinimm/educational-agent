"""Phase 0 — The Logging Skeleton.

A minimal, ugly-is-fine study loop: flashcards + photographed paper (PYP)
attempts, all tagged by concept and logged to one SQLite table. No ML
anywhere in this file — the point is to generate real interaction history
(and a pile of handwriting photos) that later phases (BKT, DKT, the CNN
grader, the RL tutor) will train and evaluate on.
"""
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

import bkt
import content
import db
import dkt
import generate
import rl_tutor
import scheduler
import tutor_policy

PHOTOS_DIR = Path(__file__).parent / "data" / "photos"

st.set_page_config(page_title="Study Agent — Phase 0", layout="centered")
db.init_db()
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)


def save_photo(uploaded_file) -> str:
    ext = Path(uploaded_file.name).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = PHOTOS_DIR / filename
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest.relative_to(Path(__file__).parent))


def photo_capture_widget(key_prefix):
    """Camera capture with a file-upload fallback. Returns an UploadedFile or None."""
    photo = st.camera_input("Snap your working", key=f"{key_prefix}_camera")
    if photo is None:
        photo = st.file_uploader(
            "...or upload an existing photo or PDF instead", type=["jpg", "jpeg", "png", "pdf"],
            key=f"{key_prefix}_upload"
        )
    return photo


def log_and_reset(concept, question_id, correct, started_at, photo_file, state_keys_to_clear):
    elapsed = round(time.time() - started_at, 1) if started_at else None
    photo_path = save_photo(photo_file) if photo_file is not None else None
    db.log_interaction(
        timestamp=datetime.now(timezone.utc).isoformat(),
        concept=concept,
        question_id=question_id,
        correct=correct,
        time_taken_seconds=elapsed,
        photo_path=photo_path,
    )
    for k in state_keys_to_clear:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------- Study ----

def page_study():
    st.header("Study")
    concepts = content.list_concepts()
    if not concepts:
        st.warning("No concepts yet. Add some on the **Manage Concepts** page first.")
        return

    mode = st.radio("Mode", ["Flashcard", "Paper / PYP attempt"], horizontal=True)

    if mode == "Flashcard":
        _study_flashcard(concepts)
    else:
        _study_paper(concepts)


def _study_flashcard(concepts):
    filter_concept = st.selectbox("Concept filter", ["All"] + concepts)
    pool = content.list_flashcards(None if filter_concept == "All" else filter_concept)
    if not pool:
        st.info("No flashcards for this filter yet. Add some on **Manage Flashcards**.")
        return

    if "card" not in st.session_state or st.button("Skip / new card"):
        st.session_state.card = random.choice(pool)
        st.session_state.show_answer = False
        st.session_state.card_start = time.time()

    card = st.session_state.card
    st.subheader(card["concept"])
    st.write(card["question"])

    if not st.session_state.get("show_answer"):
        if st.button("Show answer"):
            st.session_state.show_answer = True
            st.rerun()
    else:
        st.info(card["answer"])
        col1, col2 = st.columns(2)
        if col1.button("✅ Correct", width="stretch"):
            log_and_reset(card["concept"], card["id"], True, st.session_state.card_start, None,
                          ["card", "show_answer", "card_start"])
            st.rerun()
        if col2.button("❌ Incorrect", width="stretch"):
            log_and_reset(card["concept"], card["id"], False, st.session_state.card_start, None,
                          ["card", "show_answer", "card_start"])
            st.rerun()


def _study_paper(concepts):
    st.caption("For a question you did on paper: label it, time it, mark it, snap the working.")
    concept = st.selectbox("Concept", concepts, key="paper_concept")
    question_id = st.text_input("Question label", placeholder="e.g. 2021 Paper 1 Q5", key="paper_qid")

    if "paper_start" not in st.session_state:
        st.session_state.paper_start = time.time()
    if st.button("Restart timer"):
        st.session_state.paper_start = time.time()
    st.caption(f"Timer running: {round(time.time() - st.session_state.paper_start)}s so far")

    photo = photo_capture_widget("paper")
    correct = st.radio("Result", ["Correct", "Incorrect"], horizontal=True, key="paper_result")

    if st.button("Log attempt", type="primary"):
        if not question_id.strip():
            st.error("Give the question a label first (e.g. a paper/question number).")
        else:
            log_and_reset(concept, question_id.strip(), correct == "Correct",
                          st.session_state.paper_start, photo,
                          ["paper_start", "paper_qid", "paper_camera", "paper_upload"])
            st.success("Logged.")
            st.rerun()


# ------------------------------------------------------- Cold-start quiz ----

def page_diagnostic():
    st.header("Cold-Start Diagnostic")
    st.caption("One pass across every concept, before any model exists, so the first models "
               "aren't starting from nothing.")
    concepts = content.list_concepts()
    if not concepts:
        st.warning("No concepts yet. Add some on the **Manage Concepts** page first.")
        return

    if "diag_order" not in st.session_state:
        st.session_state.diag_order = random.sample(concepts, len(concepts))
        st.session_state.diag_idx = 0

    idx = st.session_state.diag_idx
    order = st.session_state.diag_order

    if idx >= len(order):
        st.success(f"Diagnostic complete — {len(order)} concepts covered.")
        if st.button("Run another diagnostic pass"):
            del st.session_state.diag_order
            del st.session_state.diag_idx
            st.rerun()
        return

    st.progress(idx / len(order), text=f"{idx}/{len(order)} concepts")
    concept = order[idx]
    st.subheader(concept)

    pool = content.list_flashcards(concept)
    card = pool[0] if pool else None
    if card:
        st.write(card["question"])
        if st.session_state.get("diag_show_answer"):
            st.info(card["answer"])
        elif st.button("Show answer"):
            st.session_state.diag_show_answer = True
            st.rerun()
        question_id = card["id"]
    else:
        st.info("No flashcard for this concept — attempt any question you know for it "
                "(mentally or on paper) and mark the result.")
        question_id = "diagnostic"

    if "diag_start" not in st.session_state:
        st.session_state.diag_start = time.time()

    photo = photo_capture_widget("diag")
    col1, col2 = st.columns(2)
    if col1.button("✅ Correct", width="stretch", key="diag_correct"):
        log_and_reset(concept, question_id, True, st.session_state.diag_start, photo, [])
        st.session_state.diag_idx += 1
        st.session_state.pop("diag_start", None)
        st.session_state.pop("diag_show_answer", None)
        st.rerun()
    if col2.button("❌ Incorrect", width="stretch", key="diag_incorrect"):
        log_and_reset(concept, question_id, False, st.session_state.diag_start, photo, [])
        st.session_state.diag_idx += 1
        st.session_state.pop("diag_start", None)
        st.session_state.pop("diag_show_answer", None)
        st.rerun()


# ------------------------------------------------------ Generate (LLM) ----

def page_generate():
    st.header("Generate")
    st.caption(
        "Claude drafts study material grounded in your notes/uploads and your own concept "
        "list — nothing is saved until you review and approve it."
    )
    concepts = content.list_concepts()
    if not concepts:
        st.warning("Add concepts first on **Manage Concepts** — generated material must tag one of them.")
        return

    api_key = st.sidebar.text_input(
        "OpenAI API key", type="password",
        help="Falls back to the OPENAI_API_KEY environment variable if left blank.",
    )

    mode = st.radio("Mode", ["Flashcards", "Flowchart", "Question Paper"], horizontal=True)

    if mode == "Flashcards":
        _generate_flashcards(concepts, api_key)
    elif mode == "Flowchart":
        _generate_flowchart(concepts, api_key)
    else:
        _generate_question_paper(concepts, api_key)


def _generate_flashcards(concepts, api_key):
    source = st.radio("Source", ["Upload a document", "From your notes corpus"], horizontal=True)
    n_cards = st.slider("Number of cards to draft", 1, 10, 5)

    if source == "Upload a document":
        uploaded = st.file_uploader("Notes / PYP (PDF, image, or text)", type=["pdf", "png", "jpg", "jpeg", "txt"])
        can_generate = uploaded is not None
    else:
        concept = st.selectbox("Concept", concepts, key="corpus_flashcard_concept")
        can_generate = True

    if st.button("Draft flashcards", type="primary", disabled=not can_generate):
        with st.spinner("Drafting..."):
            try:
                if source == "Upload a document":
                    cards = generate.draft_flashcards(uploaded, concepts, api_key, n_cards)
                else:
                    cards = generate.draft_flashcards_from_corpus(concept, api_key, n_cards)
                for c in cards:
                    c["include"] = True
                st.session_state.drafted_cards = cards
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

    drafted = st.session_state.get("drafted_cards")
    if not drafted:
        return

    st.subheader("Review before saving")
    edited = st.data_editor(
        drafted,
        column_config={
            "include": st.column_config.CheckboxColumn("Include"),
            "concept": st.column_config.SelectboxColumn("Concept", options=concepts),
            "question": st.column_config.TextColumn("Question", width="large"),
            "answer": st.column_config.TextColumn("Answer", width="large"),
        },
        column_order=["include", "concept", "question", "answer"],
        width="stretch",
        num_rows="fixed",
        key="drafted_editor",
    )

    if st.button("Add approved cards to flashcard bank"):
        added = 0
        for row in edited:
            if row.get("include") and row.get("question", "").strip() and row.get("answer", "").strip():
                content.add_flashcard(row["concept"], row["question"].strip(), row["answer"].strip())
                added += 1
        st.session_state.pop("drafted_cards", None)
        st.success(f"Added {added} card(s).")
        st.rerun()


def _generate_flowchart(concepts, api_key):
    concept = st.selectbox("Concept", concepts, key="flowchart_concept")
    if st.button("Draft flowchart", type="primary"):
        with st.spinner("Drafting..."):
            try:
                st.session_state.drafted_flowchart = generate.draft_flowchart(concept, api_key)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

    result = st.session_state.get("drafted_flowchart")
    if not result:
        return

    st.subheader(concept)
    st.write(result["explanation"])
    try:
        st.graphviz_chart(result["dot_source"])
    except Exception:
        st.error("Claude's diagram wasn't valid DOT syntax — here's the raw source instead.")
        st.code(result["dot_source"])


def _generate_question_paper(concepts, api_key):
    st.caption("Concepts you're weakest on (per BKT) are pre-selected — add or remove before generating.")
    default_weak = tutor_policy.weakest_concepts(min(5, len(concepts)))
    selected = st.multiselect("Concepts to cover", concepts, default=default_weak, key="paper_concepts")
    n_questions = st.slider("Number of questions", 3, 8, 5)

    if st.button("Draft question paper", type="primary", disabled=not selected):
        with st.spinner("Drafting..."):
            try:
                st.session_state.drafted_paper = generate.draft_question_paper(selected, api_key, n_questions)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                return

    paper = st.session_state.get("drafted_paper")
    if not paper:
        return

    show_answers = st.checkbox("Show mark scheme / model answers")
    st.subheader("Question Paper")
    lines = ["# O-Level Chemistry — Generated Question Paper", ""]
    for i, q in enumerate(paper, start=1):
        st.markdown(f"**Q{i}. [{q['concept']}] ({q['marks']} marks)**  \n{q['question']}")
        lines += [f"Q{i}. [{q['concept']}] ({q['marks']} marks)", "", q["question"], ""]
        if show_answers:
            st.info(q["model_answer"])
        lines += ["Model answer:", q["model_answer"], ""]

    st.download_button(
        "Download as Markdown", "\n".join(lines), file_name="question_paper.md", mime="text/markdown"
    )


# --------------------------------------------------------- Manage content ----

def page_manage_concepts():
    st.header("Manage Concepts")
    st.caption("Flat list first; hierarchy later if ever. Aim for 30-80 concepts covering your syllabus.")

    with st.form("add_concept", clear_on_submit=True):
        name = st.text_input("New concept")
        if st.form_submit_button("Add") and name.strip():
            content.add_concept(name.strip())
            st.rerun()

    concepts = content.list_concepts()
    st.write(f"**{len(concepts)} concepts**")
    for c in concepts:
        col1, col2 = st.columns([5, 1])
        col1.write(c)
        if col2.button("Remove", key=f"rm_{c}"):
            content.remove_concept(c)
            st.rerun()


def page_manage_flashcards():
    st.header("Manage Flashcards")
    concepts = content.list_concepts()
    if not concepts:
        st.warning("Add concepts first on **Manage Concepts**.")
        return

    with st.form("add_card", clear_on_submit=True):
        concept = st.selectbox("Concept", concepts)
        question = st.text_area("Question")
        answer = st.text_area("Answer")
        if st.form_submit_button("Add card") and question.strip() and answer.strip():
            content.add_flashcard(concept, question.strip(), answer.strip())
            st.rerun()

    st.divider()
    filter_concept = st.selectbox("Filter", ["All"] + concepts, key="manage_filter")
    cards = content.list_flashcards(None if filter_concept == "All" else filter_concept)
    st.write(f"**{len(cards)} cards**")
    for c in cards:
        with st.expander(f"[{c['concept']}] {c['question'][:60]}"):
            st.write(f"**Q:** {c['question']}")
            st.write(f"**A:** {c['answer']}")
            if st.button("Remove", key=f"rmcard_{c['id']}"):
                content.remove_flashcard(c["id"])
                st.rerun()


# ---------------------------------------------------- Knowledge Tracing ----

def page_knowledge_tracing():
    st.header("Knowledge Tracing (BKT)")
    st.caption(
        "A from-scratch Bayesian Knowledge Tracing model, fit per concept from your "
        "interaction log. P(known) is the model's current belief you've mastered a "
        "concept; AUC is how well it predicts your next answer — the bar any later "
        "model (DKT) has to beat."
    )
    results = bkt.evaluate_all_concepts()
    if not results:
        st.info("Not enough interactions yet — need at least 3 per concept. "
                 "Run `python seed_data.py` for demo data, or go study something.")
        return

    rows = []
    for concept, r in results.items():
        rows.append({
            "concept": concept,
            "attempts": r["n"],
            "P(known)": round(r["final_p_known"], 3),
            "naive_accuracy": round(r["naive_accuracy"], 3),
            "AUC": round(r["auc"], 3) if r["auc"] is not None else None,
        })
    df = pd.DataFrame(rows).sort_values("P(known)")

    aucs = df["AUC"].dropna()
    col1, col2, col3 = st.columns(3)
    col1.metric("Concepts modeled", len(df))
    col2.metric("Mean BKT AUC", f"{aucs.mean():.3f}" if len(aucs) else "n/a")
    col3.metric("Concepts below 60% mastery", int((df["P(known)"] < 0.6).sum()))

    st.subheader("P(known) by concept")
    st.bar_chart(df.set_index("concept")["P(known)"])

    st.subheader("Per-concept detail")
    st.dataframe(df, width="stretch", hide_index=True)


# -------------------------------------------------------- Review Queue ----

def page_review_queue():
    st.header("Review Queue")
    st.caption(
        "Predicted recall decays from each concept's BKT P(known) the longer it's "
        "been since you last practiced it. Concepts are flagged once predicted "
        f"recall drops to {int(scheduler.REVIEW_THRESHOLD * 100)}% — review before you actually forget."
    )
    queue = scheduler.review_queue()
    if not queue:
        st.info("Not enough data yet — need at least 3 interactions per concept for BKT to fit it.")
        return

    df = pd.DataFrame(queue)
    due = df[df["due"]]

    col1, col2 = st.columns(2)
    col1.metric("Concepts due for review", len(due))
    col2.metric("Concepts tracked", len(df))

    st.subheader("Due now, most urgent first")
    if due.empty:
        st.success("Nothing due — you're caught up.")
    else:
        st.dataframe(
            due[["concept", "p_known", "days_since_review", "predicted_recall"]],
            width="stretch", hide_index=True,
        )

    st.subheader("Predicted recall, all tracked concepts")
    st.bar_chart(df.set_index("concept")["predicted_recall"])


# ---------------------------------------------------- Model Comparison ----

def page_model_comparison():
    st.header("Model Comparison")
    st.caption(
        "Naive per-concept accuracy vs BKT vs DKT. BKT's AUC is the mean of each "
        "concept's own causal next-answer prediction across its full history; DKT's "
        "AUC is on a held-out chronological tail of the whole log (last ~20%, never "
        "shuffled) — the evaluation windows differ slightly, noted here for honesty "
        "rather than presented as identical."
    )
    bkt_results = bkt.evaluate_all_concepts()
    if not bkt_results:
        st.info("Not enough data yet — need at least 3 interactions per concept.")
        return

    bkt_aucs = [r["auc"] for r in bkt_results.values() if r["auc"] is not None]
    bkt_mean_auc = sum(bkt_aucs) / len(bkt_aucs) if bkt_aucs else None
    naive_mean = sum(r["naive_accuracy"] for r in bkt_results.values()) / len(bkt_results)

    if st.button("Train DKT and evaluate", type="primary"):
        with st.spinner("Training a small LSTM on your interaction log..."):
            try:
                st.session_state.dkt_result = dkt.train_and_evaluate()
            except Exception as e:
                st.error(f"DKT training failed: {e}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Naive accuracy (mean)", f"{naive_mean:.3f}")
    col2.metric("BKT mean AUC", f"{bkt_mean_auc:.3f}" if bkt_mean_auc is not None else "n/a")
    dkt_result = st.session_state.get("dkt_result")
    col3.metric("DKT held-out AUC",
                f"{dkt_result['auc']:.3f}" if dkt_result and dkt_result["auc"] is not None else "not trained yet")

    if dkt_result:
        st.caption(
            f"Trained on the first {dkt_result['n_train']} interactions; evaluated on the "
            f"trailing {dkt_result['n_test']} (naive accuracy on that same held-out slice: "
            f"{dkt_result['naive_accuracy']:.3f})."
        )


# -------------------------------------------------------- Tutor Policy ----

def page_tutor_policy():
    st.header("What to Study Next")
    st.caption(
        "A lightweight adaptive policy — not the guide's full RL tutor (that needs a "
        "trustworthy DKT to simulate against), but a real mastery/urgency/exploration-"
        "driven scheduler that adapts as your interaction log grows."
    )
    if st.button("Recommend a concept", type="primary"):
        st.session_state.tutor_pick = tutor_policy.recommend_next()

    pick = st.session_state.get("tutor_pick")
    if not pick:
        return

    st.subheader(pick["concept"])
    st.write(f"**Why:** {pick['reason']}")
    st.metric("Priority score", f"{pick['score']:.3f}")

    st.subheader("All concept scores")
    scores_df = pd.DataFrame(
        sorted(pick["all_scores"].items(), key=lambda kv: -kv[1]), columns=["concept", "score"]
    )
    st.bar_chart(scores_df.set_index("concept")["score"])


# ------------------------------------------------------------ RL Tutor ----

def page_rl_tutor():
    st.header("RL Tutor")
    st.caption(
        "A real Q-learning agent (small DQN), trained inside a simulator built from each "
        "concept's own fitted BKT parameters — real interaction data (a few hundred rows) "
        "isn't enough to train an RL agent directly, so BKT's P(L0)/P(T)/P(S)/P(G) act as a "
        "generative model of a student answering and learning, exactly the 'model of your "
        "own learning' the RL tutor is meant to train inside. State = P(known) per concept; "
        "action = which concept to study next; reward = the mastery gain from that step."
    )

    if st.button("Train RL tutor", type="primary"):
        with st.spinner("Training a DQN against the BKT simulator..."):
            try:
                q_net, env = rl_tutor.train_dqn()
                st.session_state.rl_qnet = q_net
                st.session_state.rl_env = env
                st.session_state.rl_eval = rl_tutor.evaluate(q_net, env)
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

    q_net = st.session_state.get("rl_qnet")
    env = st.session_state.get("rl_env")
    eval_result = st.session_state.get("rl_eval")
    if not q_net:
        return

    st.subheader("Policy comparison (simulated study sessions)")
    st.caption(
        "Average final mean P(known) across concepts after a 20-step simulated session, "
        "higher is better. 'Greedy weakest-first' can underperform random — it gets stuck "
        "grinding on low-learn-rate concepts instead of spreading effort where it pays off."
    )
    comp_df = pd.DataFrame(
        [("Random", eval_result["random"]),
         ("Greedy weakest-first", eval_result["greedy_weakest"]),
         ("RL agent (trained)", eval_result["rl_agent"])],
        columns=["policy", "avg_final_mastery"],
    )
    st.bar_chart(comp_df.set_index("policy")["avg_final_mastery"])
    st.dataframe(comp_df, width="stretch", hide_index=True)

    st.subheader("What the trained agent recommends right now")
    concept, q_values = rl_tutor.recommend_next_real(q_net, env.concepts)
    st.write(f"**{concept}**")
    q_df = pd.DataFrame({"concept": env.concepts, "Q-value": q_values})
    st.bar_chart(q_df.set_index("concept")["Q-value"])


# --------------------------------------------------------------- Stats ----

def page_stats():
    st.header("Stats")
    rows = db.all_interactions()
    if not rows:
        st.info("No interactions logged yet — go study something.")
        return

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    col1, col2, col3 = st.columns(3)
    col1.metric("Total interactions", len(df))
    col2.metric("Photos captured", df["photo_path"].notna().sum())
    col3.metric("Days active", df["date"].nunique())

    st.subheader("Naive per-concept accuracy")
    acc = pd.DataFrame(db.concept_accuracy())
    acc["accuracy"] = (acc["correct"] / acc["attempts"]).round(3)
    st.dataframe(acc[["concept", "attempts", "accuracy", "avg_time_seconds"]], width="stretch")
    st.bar_chart(acc.set_index("concept")["accuracy"])

    with st.expander("Raw log"):
        st.dataframe(df, width="stretch")


# ---------------------------------------------------------------- Nav ----

PAGES = {
    "Study": page_study,
    "Cold-Start Diagnostic": page_diagnostic,
    "Generate": page_generate,
    "Knowledge Tracing": page_knowledge_tracing,
    "Review Queue": page_review_queue,
    "Model Comparison": page_model_comparison,
    "What to Study Next": page_tutor_policy,
    "RL Tutor": page_rl_tutor,
    "Manage Concepts": page_manage_concepts,
    "Manage Flashcards": page_manage_flashcards,
    "Stats": page_stats,
}

st.sidebar.title("Study Agent")
st.sidebar.caption("Phase 0 — logging skeleton, no ML")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
