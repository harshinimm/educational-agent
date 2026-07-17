"""Hand-rolled retrieval: TF-IDF vectorization + our own cosine-similarity
ranking, concept-filtered first. Not a vector DB or an off-the-shelf RAG
library — just enough retrieval to ground the LLM layer in the user's own
material, per the guide's architecture (concept-filtered + embedding-ranked
search over the user's corpus).
"""
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import content

NOTES_PATH = Path(__file__).parent / "data" / "notes.csv"


def _load_chunks(concept):
    chunks = []
    if NOTES_PATH.exists():
        with open(NOTES_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["concept"] == concept:
                    chunks.append(row["chunk_text"])
    return chunks


def _fallback_chunks(concept):
    """No notes for this concept yet — fall back to its flashcard Q/A pairs."""
    cards = content.list_flashcards(concept)
    return [f"{c['question']} {c['answer']}" for c in cards]


def retrieve(concept, query, k=3):
    """Top-k text chunks for `concept`, ranked by cosine similarity to `query`."""
    chunks = _load_chunks(concept) or _fallback_chunks(concept)
    if not chunks:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1])[0]
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [text for _, text in ranked[:k]]
