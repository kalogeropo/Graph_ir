"""Mine query termsets using the collection's document-frequency postings."""

from itertools import combinations


def intersection(a: set, b: set) -> set:
    """Return IDs shared by two document sets."""
    return a & b


def create_candidate_1(query, inv_index):
    """Build unique singleton candidates in query order.

    Query must contain tokens. Posting dictionaries map original document IDs
    to term frequencies; only their keys are needed for support counting.
    """
    if isinstance(query, str):
        raise TypeError("query must be a sequence of tokens; tokenize raw text first")

    one_termsets = {}
    for term in query:
        termset = frozenset((term,))
        if term in inv_index and termset not in one_termsets:
            one_termsets[termset] = set(inv_index[term]["posting_list"])

    return one_termsets


def create_freq_term(termsets, min_freq):
    """Keep termsets occurring in at least min_freq distinct documents."""
    return {
        termset: document_ids
        for termset, document_ids in termsets.items()
        if len(document_ids) >= min_freq
    }


def create_candidate_k(freq_termsets, k):
    """Join frequent termsets and prune candidates with infrequent subsets.

    k=0 joins singletons, k=1 joins pairs, and so on. Document IDs remain sets
    throughout mining to avoid rebuilding sets for every intersection.
    """
    candidates = {}
    for first, second in combinations(freq_termsets, 2):
        if len(first & second) != k:
            continue

        termset = first | second
        if termset in candidates:
            continue
        if k > 0 and any(
            termset - {term} not in freq_termsets for term in termset
        ):
            continue

        candidates[termset] = intersection(
            freq_termsets[first], freq_termsets[second]
        )

    return candidates


def apriori(query, inv_index, min_freq: int):
    """Return frequent query termsets mapped to sets of original document IDs.

    min_freq is a positive document-count threshold, not a term-frequency
    threshold. Results follow level order, starting with singleton
    candidates in query order. Repeated and unknown query terms add no candidates.

    Raises:
        TypeError: If query is raw text instead of tokens.
        ValueError: If min_freq is not a positive integer.
    """
    if isinstance(min_freq, bool) or not isinstance(min_freq, int) or min_freq < 1:
        raise ValueError("min_freq must be a positive integer")

    frequent = create_freq_term(create_candidate_1(query, inv_index), min_freq)
    termsets = {}
    k = 0
    while frequent:
        termsets.update(frequent)
        candidates = create_candidate_k(frequent, k)
        frequent = create_freq_term(candidates, min_freq)
        k += 1
    return termsets
