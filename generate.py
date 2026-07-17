"""LLM generation layer — thin and last, grounded in retrieved material.

Uses the OpenAI API (switched from Anthropic per the hackathon's
expectations). Three modes, all requesting structured JSON output so
results validate cleanly against the app's data model:
  - Flashcards: from an uploaded document, or from the retrieved notes corpus.
  - Flowchart: a process/mechanism diagram (Graphviz DOT) for one concept.
  - Question Paper: see tutor_policy.py for the weak-area concept selection
    this mode is grounded against.

Pulled forward from the guide's Phase 4b (LLM layer, last, thin) at the
user's request — kept as its own module so it stays a clearly separate,
swappable layer from the logging/tracing/retrieval underneath it. Function
signatures are unchanged from the Anthropic version so app.py didn't need
to change at all.
"""
import base64
import io
import json
import os

from openai import OpenAI

import retriever

MODEL = "gpt-4o"  # swap to the exact model string your hackathon expects (e.g. "gpt-5.6") here


def _client(api_key):
    return OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))


def _file_content_parts(uploaded_file):
    media_type = uploaded_file.type or ""
    raw = uploaded_file.getvalue()
    if media_type.startswith("image/"):
        data = base64.standard_b64encode(raw).decode("utf-8")
        return [{"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}]
    if media_type == "application/pdf":
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [{"type": "text", "text": text}]
    return [{"type": "text", "text": raw.decode("utf-8", errors="ignore")}]


def _grounding_text(chunks):
    joined = "\n\n".join(f"- {c}" for c in chunks)
    return f"Reference material:\n{joined}"


def _structured_call(api_key, content_parts, schema_name, schema):
    response = _client(api_key).chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content_parts}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
    )
    return json.loads(response.choices[0].message.content)


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
    content_parts = _file_content_parts(uploaded_file) + [{"type": "text", "text": prompt}]
    data = _structured_call(api_key, content_parts, "flashcards", _flashcard_schema(concepts))
    return data["cards"]


def draft_flashcards_from_corpus(concept, api_key, n_cards=5):
    """Draft flashcards for one concept, grounded in the retrieved notes corpus."""
    chunks = retriever.retrieve(concept, concept, k=5)
    if not chunks:
        raise ValueError(f"No notes or flashcards found for '{concept}' to ground generation on.")

    prompt = (
        f"{_grounding_text(chunks)}\n\n"
        f"Draft up to {n_cards} flashcards about the concept \"{concept}\" using ONLY the "
        "reference material above. Tag every card with this exact concept name."
    )
    data = _structured_call(
        api_key, [{"type": "text", "text": prompt}], "flashcards", _flashcard_schema([concept])
    )
    return data["cards"]


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
        f"{_grounding_text(chunks)}\n\n"
        f"Using ONLY the reference material above, produce a step-by-step process/mechanism "
        f"flowchart for the concept \"{concept}\". Return a short explanation, and separately a "
        "Graphviz DOT digraph (valid DOT syntax, e.g. 'digraph G { A -> B; }') representing the "
        "steps as nodes and arrows. Keep it to 4-8 nodes."
    )
    return _structured_call(api_key, [{"type": "text", "text": prompt}], "flowchart", _FLOWCHART_SCHEMA)


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
        f"{_grounding_text(grounding_chunks)}\n\n"
        f"Draft a {n_questions}-question mock exam paper covering these concepts: "
        f"{', '.join(concepts)}. Use ONLY the reference material above as grounding, spread "
        "questions roughly evenly across the listed concepts, tag each with its exact concept "
        "name, assign sensible marks (2-6), and give a concise model answer for each."
    )
    schema = json.loads(json.dumps(_QUESTION_PAPER_SCHEMA_TEMPLATE))
    schema["properties"]["questions"]["items"]["properties"]["concept"]["enum"] = concepts

    data = _structured_call(api_key, [{"type": "text", "text": prompt}], "question_paper", schema)
    return data["questions"]
