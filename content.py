"""CSV-backed storage for the concept map and the manual flashcard bank.

Kept as plain CSV (not SQLite) because these are hand-edited content, not
the interaction log — easy to open in a spreadsheet, easy to diff in git.
"""
import csv
import uuid
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CONCEPTS_PATH = DATA_DIR / "concepts.csv"
FLASHCARDS_PATH = DATA_DIR / "flashcards.csv"


def _ensure_files():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONCEPTS_PATH.exists():
        with open(CONCEPTS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["concept"])
    if not FLASHCARDS_PATH.exists():
        with open(FLASHCARDS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["id", "concept", "question", "answer"])


def list_concepts():
    _ensure_files()
    with open(CONCEPTS_PATH, encoding="utf-8") as f:
        return [row["concept"] for row in csv.DictReader(f) if row["concept"].strip()]


def add_concept(name):
    name = name.strip()
    if not name:
        return
    existing = list_concepts()
    if name in existing:
        return
    _ensure_files()
    with open(CONCEPTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([name])


def remove_concept(name):
    _ensure_files()
    remaining = [c for c in list_concepts() if c != name]
    with open(CONCEPTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept"])
        for c in remaining:
            w.writerow([c])


def list_flashcards(concept=None):
    _ensure_files()
    with open(FLASHCARDS_PATH, encoding="utf-8") as f:
        cards = list(csv.DictReader(f))
    if concept:
        cards = [c for c in cards if c["concept"] == concept]
    return cards


def add_flashcard(concept, question, answer):
    _ensure_files()
    card_id = uuid.uuid4().hex[:8]
    with open(FLASHCARDS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([card_id, concept, question, answer])
    return card_id


def remove_flashcard(card_id):
    _ensure_files()
    remaining = [c for c in list_flashcards() if c["id"] != card_id]
    with open(FLASHCARDS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "concept", "question", "answer"])
        for c in remaining:
            w.writerow([c["id"], c["concept"], c["question"], c["answer"]])
