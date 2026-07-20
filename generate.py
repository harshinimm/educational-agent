"""LLM generation layer — thin and last, grounded in retrieved material.

Three modes, all calling Claude with structured output so results validate
cleanly against the app's data model:
  - Flashcards: from an uploaded document, or from the retrieved notes corpus.
  - Flowchart: a process/mechanism diagram (Graphviz DOT) for one concept.
  - Question Paper: see tutor_policy.py for the weak-area concept selection
    this mode is grounded against.

Pulled forward from the guide's Phase 4b (LLM layer, last, thin) at the
user's request — kept as its own module so it stays a clearly separate,
swappable layer from the logging/tracing/retrieval underneath it.
"""
import base64
import json
import os

import anthropic

import retriever

MODEL = "claude-opus-4-8"


def _client(api_key):
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


def _file_content_block(uploaded_file):
    media_type = uploaded_file.type or ""
    raw = uploaded_file.getvalue()
    if media_type.startswith("image/"):
        data = base64.standard_b64encode(raw).decode("utf-8")
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    if media_type == "application/pdf":
        data = base64.standard_b64encode(raw).decode("utf-8")
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}
    return {"type": "text", "text": raw.decode("utf-8", errors="ignore")}


def _grounding_text_block(chunks):
    joined = "\n\n".join(f"- {c}" for c in chunks)
    return {"type": "text", "text": f"Reference material:\n{joined}"}


# ------------------------------------------------------------ Flashcards ----

_FLASHCARD_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["concept", "question", "answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["cards"],
    "additionalProperties": False,
}


def _flashcard_schema(concepts):
    schema = json.loads(json.dumps(_FLASHCARD_SCHEMA_TEMPLATE))
    schema["properties"]["cards"]["items"]["properties"]["concept"]["enum"] = concepts
    return schema


def draft_flashcards(uploaded_file, concepts, api_key, n_cards=5):
    """Draft flashcards from an uploaded document (PDF/image/text)."""
    prompt = (
        f"Draft up to {n_cards} flashcards from the attached study material. "
        "Each card must be tagged with exactly one concept from this fixed list "
        "(pick the closest match, never invent a new concept): "
        f"{', '.join(concepts)}. "
        "Questions should be answerable from the material; keep answers concise."
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": _flashcard_schema(concepts)}},
        messages=[{"role": "user", "content": [_file_content_block(uploaded_file), {"type": "text", "text": prompt}]}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["cards"]


def draft_flashcards_from_corpus(concept, api_key, n_cards=5):
    """Draft flashcards for one concept, grounded in the retrieved notes corpus."""
    chunks = retriever.retrieve(concept, concept, k=5)
    if not chunks:
        raise ValueError(f"No notes or flashcards found for '{concept}' to ground generation on.")

    prompt = (
        f"Draft up to {n_cards} flashcards about the concept \"{concept}\" using ONLY the "
        "attached reference material. Tag every card with this exact concept name."
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": _flashcard_schema([concept])}},
        messages=[{"role": "user", "content": [_grounding_text_block(chunks), {"type": "text", "text": prompt}]}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["cards"]


# ------------------------------------------------------------- Flowchart ----

_FLOWCHART_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "dot_source": {"type": "string"},
    },
    "required": ["explanation", "dot_source"],
    "additionalProperties": False,
}


def draft_flowchart(concept, api_key):
    """Return {"explanation", "dot_source"} — a Graphviz DOT process/mechanism
    diagram for `concept`, grounded in the retrieved notes corpus."""
    chunks = retriever.retrieve(concept, concept, k=5)
    if not chunks:
        raise ValueError(f"No notes or flashcards found for '{concept}' to ground generation on.")

    prompt = (
        f"Using ONLY the attached reference material, produce a step-by-step process/mechanism "
        f"flowchart for the concept \"{concept}\". Return a short explanation, and separately a "
        "Graphviz DOT digraph (valid DOT syntax, e.g. 'digraph G { A -> B; }') representing the "
        "steps as nodes and arrows. Keep it to 4-8 nodes."
    )
    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=2048,
        output_config={"format": {"type": "json_schema", "schema": _FLOWCHART_SCHEMA}},
        messages=[{"role": "user", "content": [_grounding_text_block(chunks), {"type": "text", "text": prompt}]}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# --------------------------------------------------------- Question Paper ----

_QUESTION_PAPER_SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "question": {"type": "string"},
                    "marks": {"type": "integer"},
                    "model_answer": {"type": "string"},
                },
                "required": ["concept", "question", "marks", "model_answer"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def draft_question_paper(concepts, api_key, n_questions=5):
    """Draft a mock exam paper spread across `concepts`, grounded in the
    retrieved notes corpus for each. `concepts` is typically the tutor
    policy's weak-area pick, plus whatever the user added on top."""
    grounding_chunks = []
    for c in concepts:
        for chunk in retriever.retrieve(c, c, k=3):
            grounding_chunks.append(f"[{c}] {chunk}")
    if not grounding_chunks:
        raise ValueError("No notes or flashcards found for the selected concepts to ground generation on.")

    prompt = (
        f"Draft a {n_questions}-question mock exam paper covering these concepts: "
        f"{', '.join(concepts)}. Use ONLY the attached reference material as grounding, spread "
        "questions roughly evenly across the listed concepts, tag each with its exact concept "
        "name, assign sensible marks (2-6), and give a concise model answer for each."
    )
    schema = json.loads(json.dumps(_QUESTION_PAPER_SCHEMA_TEMPLATE))
    schema["properties"]["questions"]["items"]["properties"]["concept"]["enum"] = concepts

    response = _client(api_key).messages.create(
        model=MODEL,
        max_tokens=4096,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [_grounding_text_block(grounding_chunks), {"type": "text", "text": prompt}]}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)["questions"]
