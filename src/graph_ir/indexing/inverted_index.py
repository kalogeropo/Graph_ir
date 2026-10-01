"""Build inverted indexes mapping terms to document postings."""

from collections import Counter
import json
from pathlib import Path


def build_inverted_index(documents):
    """Build an index whose postings map document IDs to term frequencies.

    Args:
        documents: List of Document objects with .terms and .id attributes.

    Returns:
        Dictionary mapping terms to records with 'id', 'term', 'total_tf', 'posting_list'.
    """
    index = {}

    for document in documents:
        for term, frequency in Counter(document.terms).items():
            if term not in index:
                index[term] = {
                    "id": len(index),
                    "term": term,
                    "total_tf": 0,
                    "posting_list": {},
                }

            index[term]["total_tf"] += frequency
            index[term]["posting_list"][document.id] = frequency

    return index


def export_inverted_index(index: dict, path: str | Path | None = None) -> list[dict]:
    """Flatten an inverted index into JSON-serializable records.

    Each record starts from {"id", "term", "total_tf", "postings"}, where
    "postings" is a list of {"doc_id", "tf"} objects rather than the raw
    posting_list dict, so document IDs of any JSON-safe type serialize
    unambiguously. Any other field present on an entry (for example, a
    model-specific weight some other code left on the index) is carried
    through as-is, so an export reflects exactly what is on the index,
    including anything that should not be there. Records follow the
    index's existing term order (its insertion order, i.e. term id order),
    so the output is deterministic without extra sorting.

    Args:
        index: An inverted index as built by build_inverted_index(), mapping
            term -> {"id", "term", "total_tf", "posting_list", ...}.
        path: If given, also write the records as JSON to this file, creating
            parent directories as needed.

    Returns:
        The list of records, always, regardless of whether path was given —
        so the same call works for in-memory inspection (e.g. in a notebook)
        and for writing an artifact.
    """
    known_fields = {"id", "term", "total_tf", "posting_list"}

    records = []
    for entry in index.values():
        record = {
            "id": entry["id"],
            "term": entry["term"],
            "total_tf": entry["total_tf"],
            "postings": [
                {"doc_id": doc_id, "tf": tf}
                for doc_id, tf in entry["posting_list"].items()
            ],
        }
        for key, value in entry.items():
            if key not in known_fields and key not in record:
                record[key] = value
        records.append(record)

    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return records
