from collections import Counter
from math import isfinite

from numpy import array, diag, fill_diagonal, ix_, mean, outer, zeros


def calc_average_edge_w(adj_matrix):
    """The legacy mean-edge-weight estimate: the matrix mean, halved.

    Not the mean over edges: it averages every cell, including the diagonal and
    the zeros, then halves it. Kept because it defines what p_val is a
    percentage *of*, so changing it would silently reinterpret every tuned
    p_val. See MERGE_PLAN rule 6.
    """
    return mean(adj_matrix) / 2


def prune_matrix(adj_matrix, threshold):
    """Zero out edges at or below threshold, in place, keeping the diagonal.

    A threshold of 1 or less is a no-op. That guard looks redundant next to the
    only caller, which already checks threshold > 1 -- but it is part of this
    function's contract, not the caller's: removing it makes
    prune_matrix(m, 1) erase every unit edge, where before it changed nothing.
    Kept so the helper stays behaviour-identical on its own terms.

    Mutates and returns the same array. Callers own a freshly built matrix, so
    copying it would just be waste. The diagonal carries each term's own
    weight, not a co-occurrence, so it survives pruning regardless of value.
    """
    diagonal = diag(adj_matrix).copy()
    if threshold > 1:
        adj_matrix[adj_matrix <= threshold] = 0
    fill_diagonal(adj_matrix, diagonal)
    return adj_matrix



def validate_window(window, name):
    """Reject a window size that is not a positive int or finite positive float.

    bool is rejected explicitly: isinstance(True, int) is True, so a flag passed
    by mistake would otherwise be read as a one-token window.
    """
    if (
        isinstance(window, bool)
        or not isinstance(window, (int, float))
        or window <= 0
        or (isinstance(window, float) and not isfinite(window))
    ):
        raise ValueError(f"{name} must be a positive integer or finite positive fraction")


def fixed_windows(tokens, window, minimum: int = 1):
    """Split tokens into consecutive, non-overlapping windows.

    An int window is a token count; a float is a fraction of len(tokens),
    rounded down and floored at `minimum`. The final window may be short.
    """
    window_size = (
        window if isinstance(window, int)
        else max(minimum, int(window * len(tokens)))
    )
    window_size = max(minimum, window_size)
    return [
        tokens[start:start + window_size]
        for start in range(0, len(tokens), window_size)
    ]


def accumulate_window_matrix(windows, terms):
    """Sum co-occurrence weights over windows of tokens into a term matrix.

    Off-diagonal entries receive frequency products between tokens that
    co-occur in the same window; diagonal entries receive tf * (tf + 1) // 2.
    Tokens outside `terms` are ignored. Windows are independent, so a term
    accumulates contributions from every window it appears in.
    """
    term_positions = {term: position for position, term in enumerate(terms)}
    adj_matrix = zeros((len(terms), len(terms)), dtype=int)
    for window_tokens in windows:
        frequencies = Counter(window_tokens)
        active_terms = [term for term in frequencies if term in term_positions]
        if not active_terms:
            continue
        positions = [term_positions[term] for term in active_terms]
        counts = array([frequencies[term] for term in active_terms], dtype=int)
        window_matrix = outer(counts, counts)
        fill_diagonal(window_matrix, counts * (counts + 1) // 2)
        adj_matrix[ix_(positions, positions)] += window_matrix
    return adj_matrix
