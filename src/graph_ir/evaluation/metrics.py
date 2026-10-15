"""Cosine ranking and retrieval metric calculations."""

import numpy as np

_TIE_TOLERANCE = 10


def cosine_similarity(query, dtm, num_docs):
    """Score each matrix column against the query vector, in column order.

    Columns must follow document order. Zero vectors receive score zero.
    Returns a (num_docs,) array, not a ranking -- see rank_scores.

    Kept separate from ranking so a caller can score a query and keep only the
    resulting (num_docs,) vector, instead of holding on to the matrix.
    """
    query = np.asarray(query, dtype=float)
    if not hasattr(dtm, "nnz"):
        dtm = np.asarray(dtm, dtype=float)
    if query.ndim != 1 or dtm.shape != (query.size, num_docs):
        raise ValueError(
            "Expected a query vector and a termset-by-document matrix "
            "with one column per document ID"
        )

    denominator = np.linalg.norm(query) * _column_norms(dtm)
    return np.divide(
        query @ dtm,
        denominator,
        out=np.zeros(num_docs),
        where=denominator != 0,
    )


def _column_norms(dtm):
    """Euclidean norm of each column, for a dense or a scipy sparse matrix."""
    if hasattr(dtm, "nnz"):
        return np.sqrt(dtm.multiply(dtm).sum(axis=0))
    return np.linalg.norm(dtm, axis=0)


def rank_scores(similarities, document_ids):
    """Order scores highest-first, returning {document_id: score}.

    Equal scores keep their original document order, and that ordering is
    reproducible: scores are compared rounded to _TIE_TOLERANCE decimals, then
    ties broken by position in document_ids. The returned values are unrounded.

    Rounding is what makes the tie an actual tie. Comparing raw floats meant any
    change to the summation order -- a different matrix layout, a sparse dot
    product, a BLAS version -- left genuinely-equal scores differing by an ULP,
    so they sorted by that noise instead of tying. On CF 18.3% of scores sit in
    a tie, and a 1-ULP perturbation reordered 10 of 25 queries.
    """
    scores = enumerate(zip(document_ids, np.asarray(similarities).tolist(), strict=True))
    return {
        document_id: score
        for _, (document_id, score) in sorted(
            scores,
            key=lambda item: (-round(item[1][1], _TIE_TOLERANCE), item[0]),
        )
    }


def evaluate_sim(query, dtm, document_ids):
    """Rank matrix columns by cosine similarity using original document IDs.

    Composes cosine_similarity and rank_scores; see those for the scoring and
    tie-breaking rules.
    """
    document_ids = list(document_ids)
    similarities = cosine_similarity(query, dtm, len(document_ids))
    return rank_scores(similarities, document_ids)


def calc_precision_recall(doc_sims, relevant, k):
    """Return mean precision/recall at relevant hits and reciprocal rank.

    These averages differ from standard precision@k, recall@k, and AP: each
    is the mean of precision/recall computed at every relevant hit's own
    rank, not a single value at rank k. k is an ordinary top-k cutoff: only
    the first k ranked documents are considered, and k=0 or a ranking
    shorter than k considers only what is actually there.
    """
    count = 0
    precision = []
    recall = []
    reciprocal_rank = 0.0

    for retrieved, document_id in enumerate(doc_sims, start=1):
        if retrieved > k:
            break
        if document_id in relevant:
            count += 1
            precision.append(count / retrieved)
            recall.append(count / len(relevant))
            if count == 1:
                reciprocal_rank = 1 / retrieved

    average_precision = sum(precision) / len(precision) if precision else 0.0
    average_recall = sum(recall) / len(recall) if recall else 0.0
    return average_precision, average_recall, reciprocal_rank
