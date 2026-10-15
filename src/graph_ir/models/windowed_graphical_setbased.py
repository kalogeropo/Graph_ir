from graph_ir.graphs.operations import fixed_windows, validate_window
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
                 h_val: int = 1,
                 k_core_bool: bool = False,
                 window_cut_off: bool = True):
        validate_window(window, "window")
        self.window = window
        self.window_cut_off = window_cut_off
        super().__init__(collection, k_core_bool=k_core_bool, h_val=h_val)

    def _window_sets(self, document):
        """Consecutive, non-overlapping windows over the document's tokens.

        Fractional sizes are rounded down, with a minimum of one token, or
        seven when window_cut_off is enabled. The final partial window is
        retained. A window spanning the whole document returns None, so
        doc_to_matrix falls back to the parent's indexed-frequency clique --
        the same matrix, computed without re-counting tokens.
        """
        minimum = 7 if self.window_cut_off else 1
        window_size = (
            self.window if isinstance(self.window, int)
            else max(minimum, int(self.window * len(document.terms)))
        )
        if max(minimum, window_size) >= len(document.terms):
            return None
        return [(1, fixed_windows(document.terms, self.window, minimum))]
