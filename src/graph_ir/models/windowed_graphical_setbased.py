from collections import Counter
from math import isfinite

from numpy import array, fill_diagonal, ix_, ndarray, outer, zeros

from graph_ir.models.graphical_setbased import GSBModel


class WindowedGSBModel(GSBModel):
    """
    The windowed version of the Graph-Based Set-Based model.
    Applies a windowed co-occurrence logic (fixed or percentage-based)
    instead of full-document adjacency.

    Args:
        collection: The document collection.
        window: Positive integer token count, or a finite positive float
            multiplied by the document length (for example, 0.1 for 10%).
        h_val: Weighting factor for enhanced edges.
        k_core_bool: Enable or disable k-core pruning.
        window_cut_off: Enforce the original minimum window size of seven
            tokens. The final incomplete window is always included.
    """

    def __init__(self,
                 collection,
                 window: int | float = 8,
                 h_val: int | float = 1,
                 k_core_bool: bool = False,
                 window_cut_off: bool = True):
        if (
            isinstance(window, bool)
            or not isinstance(window, (int, float))
            or window <= 0
            or (isinstance(window, float) and not isfinite(window))
        ):
            raise ValueError("window must be a positive integer or finite positive fraction")
        self.window = window
        self.window_cut_off = window_cut_off
        super().__init__(collection, k_core_bool=k_core_bool, h_val=h_val)

    def get_model(self):
        return self.__class__.__name__

    def doc_to_matrix(self, document, terms=None) -> ndarray:
        """Sum co-occurrence weights over consecutive, non-overlapping windows.

        Rows and columns follow the supplied terms, or their first occurrence
        order in the document. Each window contributes frequency products to
        off-diagonal entries and tf * (tf + 1) // 2 to diagonal entries.

        Fractional sizes are rounded down, with a minimum of one token. When
        window_cut_off is enabled, the minimum is seven instead. The final
        partial window is retained. Empty documents produce a (0, 0) matrix.
        The document must belong to the indexed collection; a single window
        reuses the parent's indexed frequency calculation.
        """
        if terms is None:
            terms = list(dict.fromkeys(document.terms))
        if not document.terms or not terms:
            return zeros((len(terms), len(terms)), dtype=int)

        window_size = (
            self.window if isinstance(self.window, int)
            else max(1, int(self.window * len(document.terms)))
        )
        if self.window_cut_off:
            window_size = max(7, window_size)
        if window_size >= len(document.terms):
            return super().doc_to_matrix(document, terms)

        adj_matrix = zeros((len(terms), len(terms)), dtype=int)
        term_positions = {term: position for position, term in enumerate(terms)}

        for start in range(0, len(document.terms), window_size):
            frequencies = Counter(document.terms[start:start + window_size])
            active_terms = [term for term in frequencies if term in term_positions]
            if not active_terms:
                continue
            positions = [term_positions[term] for term in active_terms]
            counts = array([frequencies[term] for term in active_terms], dtype=int)
            window_matrix = outer(counts, counts)
            fill_diagonal(window_matrix, counts * (counts + 1) // 2)
            adj_matrix[ix_(positions, positions)] += window_matrix

        return adj_matrix
