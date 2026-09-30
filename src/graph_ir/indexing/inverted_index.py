"""Build inverted indexes mapping terms to document postings."""

from collections import Counter


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
