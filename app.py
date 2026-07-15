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

import content
import db

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
            "...or upload an existing photo instead", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload"
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
        if col1.button("✅ Correct", use_container_width=True):
            log_and_reset(card["concept"], card["id"], True, st.session_state.card_start, None,
                          ["card", "show_answer", "card_start"])
            st.rerun()
        if col2.button("❌ Incorrect", use_container_width=True):
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
    if col1.button("✅ Correct", use_container_width=True, key="diag_correct"):
        log_and_reset(concept, question_id, True, st.session_state.diag_start, photo, [])
        st.session_state.diag_idx += 1
        st.session_state.pop("diag_start", None)
        st.session_state.pop("diag_show_answer", None)
        st.rerun()
    if col2.button("❌ Incorrect", use_container_width=True, key="diag_incorrect"):
        log_and_reset(concept, question_id, False, st.session_state.diag_start, photo, [])
        st.session_state.diag_idx += 1
        st.session_state.pop("diag_start", None)
        st.session_state.pop("diag_show_answer", None)
        st.rerun()


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
    st.dataframe(acc[["concept", "attempts", "accuracy", "avg_time_seconds"]], use_container_width=True)
    st.bar_chart(acc.set_index("concept")["accuracy"])

    with st.expander("Raw log"):
        st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------- Nav ----

PAGES = {
    "Study": page_study,
    "Cold-Start Diagnostic": page_diagnostic,
    "Manage Concepts": page_manage_concepts,
    "Manage Flashcards": page_manage_flashcards,
    "Stats": page_stats,
}

st.sidebar.title("Study Agent")
st.sidebar.caption("Phase 0 — logging skeleton, no ML")
choice = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[choice]()
