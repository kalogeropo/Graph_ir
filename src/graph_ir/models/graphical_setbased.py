from math import log2
from typing import Any
import warnings

from networkx import (
    Graph,
    core_number,
    from_numpy_array,
    get_node_attributes,
    selfloop_edges,
    set_node_attributes,
)
from numpy import (
    array,
    count_nonzero,
    fill_diagonal,
    flatnonzero,
    ndarray,
    outer,
    zeros,
)

from graph_ir.graphs.operations import (
    accumulate_window_matrix,
    calc_average_edge_w,
    prune_matrix,
)
from graph_ir.models.model import Model


def _is_complete(adj_matrix) -> bool:
    """True when every entry is nonzero, i.e. every term pair co-occurs.

    In a complete graph every node has the same degree, so every node shares
    the maximal core number and union_graph can take all the terms as the core
    without running a decomposition. Diagonal entries are always nonzero
    (tf * (tf + 1) / 2 >= 1 for tf >= 1), so counting all nonzeros is enough.

    Checking the matrix costs one vectorised pass over cells that were just
    built anyway, and unlike inspecting the class it stays correct for a
    windowed model whose window happens to span a whole document.
    """
    size = adj_matrix.shape[0]
    return size <= 1 or int(count_nonzero(adj_matrix)) == size * size


class GSBModel(Model):
    """The GSBModel - Graph Based extension of the SetBased model will be consisted of:
            a. graph - the union graph
            b. _nwk - the term weights derived of each node
            c. as well as any field of its superclass Model
    The main model funct and vectorizer are overridden as we need a different functionality"""

    def __init__(self,
                 collection,
                 k_core_bool: bool = False,
                 h_val: int = 1,
                 p_val: int | float = 0,
                 nwk_a: float = 1,
                 nwk_b: float = 10):
        """
        Initialize the GSBModel with optional k-core pruning and parameter normalization.

        Args:
        collection: The document collection to model.
        k_core_bool (bool): Enable k-core filtering on per-document graphs.
        h_val (int): Weight amplification factor applied to core terms, used as
            given. Must be an int: 2 means 2x, not 200%.
        p_val (int | float): Edge weight pruning threshold, as a multiple of the
            mean edge weight. An int is read as a percentage and a float as a
            plain fraction, so both p_val=80 and p_val=0.8 give 0.8.
        nwk_a, nwk_b (float): Node-weight coefficients -- a scales Wout, b
            normalizes by neighbour count (see _calculate_nwk). Exposed here
            because _calculate_nwk only ever runs from this constructor, so
            its own defaults were unreachable and could not be swept.

        Both h_val and p_val only do anything when k_core_bool is True: h scales
        the edges of terms in a document's maximal core, and pruning happens on
        the way to finding that core. Setting either without k_core_bool warns,
        rather than raising, so that a parameter sweep can pass every
        combination uniformly -- but it does warn, because the previous silence
        was indistinguishable from "this parameter does not matter".

        Raises:
            ValueError: If h_val is not an int, or p_val is not a real number.
                bool is rejected for both.
        """
        if isinstance(h_val, bool) or not isinstance(h_val, int):
            raise ValueError("h_val must be an int")
        if isinstance(p_val, bool) or not isinstance(p_val, (int, float)):
            raise ValueError("p_val must be an int percentage or a float fraction")
        if not k_core_bool:
            ineffective = [
                name for name, value, default in
                (("h_val", h_val, 1), ("p_val", p_val, 0))
                if value != default
            ]
            if ineffective:
                warnings.warn(
                    f"{' and '.join(ineffective)} has no effect when "
                    f"k_core_bool is False; pass k_core_bool=True to apply it",
                    stacklevel=2,
                )

        self.k_core_bool = k_core_bool
        self.nwk_a = nwk_a
        self.nwk_b = nwk_b
        self.h = h_val
        # An int p_val is a percentage; a float is already a fraction.
        self.p = p_val / 100 if isinstance(p_val, int) else p_val
        super().__init__(collection)

        self.graph: Graph = self.union_graph()
        self._nwk = self._calculate_nwk()

    def _model_func(self, freq_termsets: Any) -> ndarray:
        tns = zeros(len(freq_termsets), dtype=float)
        for i, termset in enumerate(freq_termsets):
            temp = 1
            for term in termset:
                if term in self._nwk:
                    temp *= self._nwk[term]
            tns[i] = round(temp, 3)
        return tns

    def _vectorizer(self, tsf_ij: ndarray, idf: ndarray, *args: Any) -> ndarray:
        """
        Applies the model-specific vectorization formula using term weighting (tns).

        Args:
            tsf_ij (np.ndarray): Termset-document frequency matrix.
            idf (np.ndarray): Inverse document frequency vector.
            *args (Any): Expects the first item to be a NumPy array `tns` of term weights.

        Returns:
            np.ndarray: The weighted document-term matrix.
        """
        tns = args[0]  # Unpack explicitly instead of `tns, *_ = args` for clarity
        return tsf_ij * (idf * tns).reshape(-1, 1)

    def union_graph(self) -> Graph:
        """Accumulate document edges without changing their insertion order.

        A complete matrix gives every term the same core number, so the whole
        term set is the maximal core and the decomposition can be skipped.
        That is read off the matrix itself (see _is_complete) rather than
        guessed from which subclass is running: a windowed model still produces
        a complete matrix when one window covers the document, and a subclass
        overriding doc_to_matrix is handled without having to know about it.
        Pruning retains diagonal weights and uses the original mean threshold.
        """
        union = Graph()

        for doc in self.collection.docs:
            terms = list(dict.fromkeys(doc.terms))
            if not terms:
                continue
            adj_matrix = self.doc_to_matrix(doc, terms)
            core_terms = set()
            pruned = False
            if self.k_core_bool:
                # Subclasses never pass p_val, so self.p is 0 for them and this
                # threshold is 0 -- no need to ask which class we are.
                threshold = self.p * calc_average_edge_w(adj_matrix)
                if threshold <= 1 and _is_complete(adj_matrix):
                    core_terms = set(terms)
                else:
                    if threshold > 1:
                        adj_matrix = prune_matrix(adj_matrix, threshold)
                        pruned = True
                    document_graph = from_numpy_array(adj_matrix)
                    document_graph.remove_edges_from(selfloop_edges(document_graph))
                    core_numbers = core_number(document_graph)
                    maximum_core = max(core_numbers.values(), default=0)
                    core_terms = {
                        terms[node] for node, number in core_numbers.items()
                        if number == maximum_core
                    }

            for i, term in enumerate(terms):
                h = self.h if term in core_terms else 1
                neighbors = union.adj[term] if term in union else {}
                row = adj_matrix[i, :i + 1]
                # A pruned row is mostly zeros, and h is always a finite int, so
                # skipping the zeros is equivalent and cheaper.
                if pruned:
                    columns = flatnonzero(row)
                    entries = zip(
                        (terms[j] for j in columns), row[columns].tolist(),
                        strict=True,
                    )
                else:
                    # row is the lower-triangle prefix, so it is shorter than
                    # terms and zip must stop at its end.
                    entries = zip(terms, row.tolist(), strict=False)

                # Native numbers make edge updates and later degree sums cheaper.
                new_edges = []
                for other, weight in entries:
                    attributes = neighbors.get(other)
                    if attributes is not None:
                        attributes["weight"] += weight * h
                    elif weight > 0:
                        new_edges.append((term, other, weight * h))
                if new_edges:
                    union.add_weighted_edges_from(new_edges)

        set_node_attributes(union, {
            term: union[term][term]["weight"] for term in union
        }, "weight")
        union.remove_edges_from(selfloop_edges(union))
        return union

    def _window_sets(self, document) -> list[tuple[float, Any]]:
        """Return the (weight, windows) pairs whose co-occurrences build a document.

        This is the hook that distinguishes the GSB variants: each supplies its
        own token windows, and doc_to_matrix combines them. Plain GSB treats the
        whole document as a single window, so it returns None here to take
        doc_to_matrix's faster indexed-frequency path instead -- the result is
        the same clique either way.
        """
        return None

    def doc_to_matrix(self, document, terms=None) -> ndarray:
        """Build the document's adjacency matrix over its distinct terms.

        Rows and columns follow the supplied terms, or first occurrence order in
        document.terms. Empty documents produce a (0, 0) matrix. Off-diagonal
        weights are frequency products and diagonal weights are tf * (tf + 1) / 2,
        summed over each window a term appears in.

        Subclasses override _window_sets rather than this method, so the terms
        defaulting and the empty-document case stay in one place.
        """
        if terms is None:
            terms = list(dict.fromkeys(document.terms))

        window_sets = self._window_sets(document)
        if window_sets is None:
            return self._whole_document_matrix(document, terms)

        if not document.terms or not terms:
            return zeros((len(terms), len(terms)), dtype=int)
        matrix = None
        for weight, windows in window_sets:
            scaled = weight * accumulate_window_matrix(windows, terms)
            matrix = scaled if matrix is None else matrix + scaled
        return matrix

    def _whole_document_matrix(self, document, terms) -> ndarray:
        """The single-window clique, from indexed frequencies.

        Equivalent to one window spanning the document, but read straight off
        the inverted index instead of re-counting tokens. The document must
        belong to the indexed collection.
        """
        inverted_index = self.collection.inverted_index
        rows = array([
            inverted_index[term]["posting_list"][document.id]
            for term in terms
        ])
        adj_matrix = outer(rows, rows)
        fill_diagonal(adj_matrix, rows * (rows + 1) * 0.5)
        return adj_matrix

    def _calculate_win(self) -> dict:
        return get_node_attributes(self.graph, 'weight')

    def _calculate_wout(self) -> dict:
        return dict(self.graph.degree(weight='weight'))  # type: ignore

    def _number_of_nbrs(self) -> dict:
        return dict(self.graph.degree())  # type: ignore

    def _calculate_nwk(self, a: float | None = None, b: float | None = None) -> dict[str, float]:
        """
        Calculate node weights (nwk) for terms in the union graph.

        Args:
            a (float): Weighting factor for Wout; defaults to self.nwk_a.
            b (float): Weighting factor for neighbor normalization; defaults
                to self.nwk_b.

        Returns:
            dict[str, float]: Mapping of term to nwk score.
        """
        a = self.nwk_a if a is None else a
        b = self.nwk_b if b is None else b
        nwk = {}
        Win = self._calculate_win()
        Wout = self._calculate_wout()
        ngb = self._number_of_nbrs()
        # Every denominator is a count plus one, so none of this can divide by
        # zero or go out of domain -- no exception handling needed.
        for term in Win:
            f = a * Wout[term] / ((Win[term] + 1) * (ngb[term] + 1))
            s = b / (ngb[term] + 1)
            nwk[term] = round(log2(1 + f) * log2(1 + s), 3)
        return nwk
