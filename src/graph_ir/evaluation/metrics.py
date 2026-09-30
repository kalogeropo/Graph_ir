"""Cosine ranking and retrieval metric calculations."""

import numpy as np


def evaluate_sim(query, dtm, document_ids):
    """Rank matrix columns by cosine similarity using original document IDs.

    Columns must follow document_ids order. Zero vectors receive score zero;
    equal scores keep their original document order.
    """
    query = np.asarray(query, dtype=float)
    dtm = np.asarray(dtm, dtype=float)
    document_ids = list(document_ids)
    if query.ndim != 1 or dtm.shape != (query.size, len(document_ids)):
        raise ValueError(
            "Expected a query vector and a termset-by-document matrix "
            "with one column per document ID"
        )

    denominator = np.linalg.norm(query) * np.linalg.norm(dtm, axis=0)
    similarities = np.divide(
        query @ dtm,
        denominator,
        out=np.zeros(len(document_ids)),
        where=denominator != 0,
    )
    scores = dict(zip(document_ids, similarities.tolist(), strict=True))
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def calc_precision_recall(doc_sims, relevant, k):
    """Return mean precision/recall at relevant hits and reciprocal rank.

    These averages differ from standard precision@k, recall@k, and AP.
    Current cutoff behavior: for k > 1, processing stops
    before rank k; k=1 does not trigger an early stop.
    """
    count = 0
    retrieved = 1
    precision = []
    recall = []
    reciprocal_rank = 0.0

    for document_id in doc_sims:
        if document_id in relevant:
            count += 1
            precision.append(count / retrieved)
            recall.append(count / len(relevant))
            if count == 1:
                reciprocal_rank = 1 / retrieved
        retrieved += 1
        if retrieved == k:
            break

    average_precision = sum(precision) / len(precision) if precision else 0.0
    average_recall = sum(recall) / len(recall) if recall else 0.0
    return average_precision, average_recall, reciprocal_rank
