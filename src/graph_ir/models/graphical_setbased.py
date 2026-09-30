from math import isfinite, log2

from networkx import Graph, set_node_attributes, get_node_attributes, core_number, selfloop_edges
from numpy import array, outer, fill_diagonal, flatnonzero, zeros, float64, ndarray
from graph_ir.models.model import Model
from graph_ir.graphs.operations import (
    calc_average_edge_w, prune_matrix, adj_to_graph,
)

from typing import Any, Dict



class GSBModel(Model):
    """The GSBModel - Graph Based extension of the SetBased model will be consisted of:
            a. graph - the union graph
            b. _nwk - the term weights derived of each node
            c. as well as any field of its superclass Model
    The main model funct and vectorizer are overridden as we need a different functionality"""

    def __init__(self,
                 collection,
                 k_core_bool: bool = False,
                 h_val: int | float = 1,
                 p_val: int | float = 0):
        """
        Initialize the GSBModel with optional k-core pruning and parameter normalization.

        Args:
        collection: The document collection to model.
        k_core_bool (bool): Enable k-core filtering on per-document graphs.
        h_val (int | float): Weight amplification factor (as int or percentage).
        p_val (int | float): Edge weight pruning threshold (as percentage or float).
        """

        self.k_core_bool = k_core_bool
        # Normalize h and p based on type
        self.h = h_val if isinstance(h_val, int) else h_val * 100
        self.p = p_val / 100 if isinstance(p_val, int) else p_val
        super().__init__(collection)

        self.model = self.get_model()
        self.graph: Graph = self.union_graph()
        self._nwk = self._calculate_nwk()

    def _model_func(self, freq_termsets: Any) -> ndarray:
        tns = zeros(len(freq_termsets), dtype=float)
        inverted_index = self.collection.inverted_index
        for i, termset in enumerate(freq_termsets):
            temp = 1
            for term in termset:
                if term in inverted_index:
                    temp *= inverted_index[term]['nwk']
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

    def get_model(self) -> str:
        return self.__class__.__name__

    def union_graph(self) -> Graph:
        """Accumulate document edges without changing their insertion order.

        Unpruned full-document matrices are cliques, so every term belongs to
        their maximal core. Overridden matrices still use core decomposition.
        Pruning retains diagonal weights and uses the original mean threshold.
        """
        union = Graph()
        standard_matrix = type(self).doc_to_matrix is GSBModel.doc_to_matrix

        for doc in self.collection.docs:
            terms = list(dict.fromkeys(doc.terms))
            if not terms:
                continue
            adj_matrix = self.doc_to_matrix(doc, terms)
            core_terms = set()
            pruned = False
            if self.k_core_bool:
                threshold = (
                    self.p * calc_average_edge_w(adj_matrix)
                    if self.model == "GSBModel" else 0
                )
                if standard_matrix and threshold <= 1:
                    core_terms = set(terms)
                else:
                    if self.model == "GSBModel" and threshold > 1:
                        adj_matrix = prune_matrix(adj_matrix, threshold)
                        pruned = True
                    document_graph = adj_to_graph(adj_matrix)
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
                if pruned and isfinite(h):
                    columns = flatnonzero(row)
                    entries = zip(
                        (terms[j] for j in columns), row[columns].tolist(),
                    )
                else:
                    entries = zip(terms, row.tolist())

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

    def doc_to_matrix(self, document, terms=None) -> ndarray:
        """Build a GSB adjacency matrix using indexed document frequencies.

        The document must belong to the indexed collection. Rows and columns
        follow the supplied terms, or first occurrence order in document.terms.
        Off-diagonal weights are frequency products; diagonal weights are
        tf * (tf + 1) / 2. Empty documents produce a (0, 0) matrix.
        """
        if terms is None:
            terms = list(dict.fromkeys(document.terms))
        inverted_index = self.collection.inverted_index
        rows = array([
            inverted_index[term]["posting_list"][document.id]
            for term in terms
        ])
        adj_matrix = outer(rows, rows)
        fill_diagonal(adj_matrix, rows * (rows + 1) * 0.5)
        return adj_matrix

    def _calculate_win(self) -> Dict:
        return get_node_attributes(self.graph, 'weight')

    def _calculate_wout(self) -> Dict[Any, Any]:
        return {node: val for (node, val) in self.graph.degree(weight='weight')}  # type: ignore

    def _number_of_nbrs(self) -> Dict:
        return {node: val for (node, val) in self.graph.degree()}  # type: ignore

    def _calculate_nwk(self, a: float = 1, b: float = 10) -> Dict[str, float]:
        """
        Calculate node weights (nwk) for terms in the union graph.

        Args:
            a (float): Weighting factor for Wout.
            b (float): Weighting factor for neighbor normalization.

        Returns:
            Dict[str, float]: Mapping of term to nwk score.
        """
        nwk = {}
        Win = self._calculate_win()
        Wout = self._calculate_wout()
        ngb = self._number_of_nbrs()
        for term in list(Win.keys()):
            try:

                f = float64(a * Wout[term] / ((Win[term] + 1) * (ngb[term] + 1)))
                s = float64(b / (ngb[term] + 1))
                score = round(log2(1 + f) * log2(1 + s), 3)

            except (ValueError, ZeroDivisionError) as e:
                print(f"Error calculating nwk for term '{term}': {e}")
                score = 0

            nwk[term] = score
            self.collection.inverted_index[term]['nwk'] = score
        return nwk
