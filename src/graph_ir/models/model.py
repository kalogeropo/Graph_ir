from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy import array
from scipy.sparse import csr_array

from graph_ir.data.collection import Collection
from graph_ir.data.preprocessing.text import remove_stopwords
from graph_ir.evaluation.metrics import (
    calc_precision_recall,
    cosine_similarity,
    rank_scores,
)
from graph_ir.models.apriori import apriori


class Model(ABC):
    def __init__(self,collection: Collection):
        self.collection = collection

        self._queryVectors = []
        # Per-query cosine scores in document order, one (num_docs,) array each.
        # fit() deliberately does NOT keep the termset-by-document matrices that
        # produce these: each one is used exactly once, and retaining all of them
        # cost 4.8 GB on CF (1,239 docs, 100 queries) against 0.99 MB for the
        # scores. See fit().
        self._docScores = []
        self._termset_counts = []
        self._weights = []
        self._query_ids = []
        self._document_ids = []
        # metrics
        self.precision = []
        self.recall = []
        self.reciprocal_ranks = []
        # model_document_ranking
        self.ranking = []

    def get_model(self) -> str:
        """This model's name, which is its class name.

        Concrete, not abstract: every subclass returned exactly this, so there
        was nothing for them to decide. Override it only to report a name that
        differs from the class name.
        """
        return self.__class__.__name__

    @property
    def model(self) -> str:
        """Alias for get_model(), kept so `model.model` still reads a name."""
        return self.get_model()

    @abstractmethod
    def _model_func(self, freq_termsets: Any) -> np.ndarray:
        pass

    # The base fit() implements the set-based document/query representation.
    # Subclasses override this to change how it is weighted.
    @abstractmethod
    def _vectorizer(self, termset_freq_ij: np.ndarray, idf: np.ndarray, *args: Any) -> np.ndarray:
        """
        Transform the raw document-term matrix into a model-specific representation.

        Args:
            termset_freq_ij (np.ndarray): Termset-document frequency matrix.
            idf (np.ndarray): Inverse document frequency vector.
            *args (Any): Additional arguments such as model-specific weights.

        Returns:
            np.ndarray: Final document-term matrix or score matrix.
        """

    def calculate_tsf(self, termsets: dict) -> csr_array:
        """Build a termset-by-document matrix using direct frequency lookups.

        termsets maps each nonempty termset to the IDs of documents containing all
        its terms, as produced by Apriori. Rows follow termset insertion order;
        columns follow collection.docs order. Frequencies use logarithmic weighting
        and rounding: round(1 + log2(minimum term frequency), 3).

        Returned sparse, because the matrix is overwhelmingly empty: a termset
        has an entry only for the documents containing all of its terms, which
        across CF's 100 queries is 6,309,869 entries in 484,994 x 1,239 cells --
        1.05% density, and 921 MB dense for the largest single query against
        3.5 MB sparse. Values are identical either way; no weight can be zero,
        since round(1 + log2(f), 3) >= 1.0 for every f >= 1, so nothing real is
        lost to the sparsity. Use termset_matrix() for a dense one to inspect.
        """
        frequency_weights = {}
        rows, columns, weights = [], [], []

        for row, (termset, document_ids) in enumerate(termsets.items()):
            frequencies = [self.collection.inverted_index[term]["posting_list"] for term in termset]
            # document_ids is a set, so each (row, column) is emitted once --
            # COO construction sums duplicates rather than overwriting them.
            for document_id in document_ids:
                frequency = min(lookup[document_id] for lookup in frequencies)
                weight = frequency_weights.get(frequency)
                if weight is None:
                    weight = round(1 + np.log2(frequency), 3)
                    frequency_weights[frequency] = weight
                rows.append(row)
                columns.append(self.collection.doc_id_to_position[document_id])
                weights.append(weight)

        return csr_array(
            (weights, (rows, columns)),
            shape=(len(termsets), self.collection.num_docs),
            dtype=float,
        )
    def calculate_ts_idf(self, termsets):
        """Inverse document frequency per termset, in termset order.

        len(document_ids) is in how many documents each termset appears.
        """
        num_docs = self.collection.num_docs
        return array([
            round(np.log2(1 + (num_docs / len(document_ids))), 3)
            for document_ids in termsets.values()
        ])

    def mine_termsets(self, query, min_freq: int = 1, stopwords: bool = False):
        """Normalize one query the way fit() does, then mine its termsets.

        Applies the collection's preprocessor (or whitespace splitting), the
        optional stopword filter, and Apriori, so a single query can be
        inspected under exactly the rules fit() would use.
        """
        if self.collection.preprocessor is not None:
            query = self.collection.preprocessor(query)
        elif isinstance(query, str):
            query = query.split()
        if stopwords:
            query = remove_stopwords(
                query, self.collection.stopwords, case_insensitive=True,
            )
        return apriori(query, self.collection.inverted_index, min_freq)

    def termset_matrix(self, query, min_freq: int = 1, stopwords: bool = False):
        """Build one query's termset-by-document matrix densely, on demand.

        fit() releases these as it goes (see fit()), so this rebuilds a single
        one for inspection or debugging without holding every query's matrix.
        Dense, unlike calculate_tsf, because this exists to be looked at -- so
        do not call it on a query with a large termset count.
        """
        return self.calculate_tsf(
            self.mine_termsets(query, min_freq, stopwords)
        ).toarray()

    def fit(self, queries=None, min_freq: int = 1, stopwords: bool = False):
        """Score every query, keeping its identity and its document scores.

        Query mappings carry explicit IDs. Tokenized query lists use collection
        query order when their lengths match; subsets need an ID mapping for
        evaluation. When the collection has a preprocessor, the same rules are
        applied to raw and tokenized queries. Otherwise, raw strings are split
        on whitespace. Optional query stopword removal follows tokenization
        and is case-insensitive against collection.stopwords regardless of
        whether a preprocessor already lowercased the tokens.

        Each query is scored in the same pass that builds its
        termset-by-document matrix, and only the resulting (num_docs,) score
        vector is kept -- the matrix is released immediately. It is needed
        exactly once, so retaining every query's matrix cost 4.8 GB on CF
        against 0.99 MB for the scores, and grows with
        documents x termsets x queries. evaluate() turns the stored scores into
        rankings and metrics.
        """
        if queries is None:
            queries = self.collection.queries
        if isinstance(queries, dict):
            query_items = [(str(query_id), query) for query_id, query in queries.items()]
        else:
            query_ids = [str(query_id) for query_id in self.collection.queries]
            if len(queries) != len(query_ids):
                query_ids = [None] * len(queries)
            query_items = list(zip(query_ids, queries, strict=True))

        for values in (
            self._query_ids, self._queryVectors, self._docScores,
            self._termset_counts, self._weights,
            self.precision, self.recall, self.reciprocal_ranks, self.ranking,
        ):
            values.clear()

        # Snapshot document identity now: calculate_tsf lays out matrix columns
        # using the collection as it is at this moment. evaluate() must label
        # those same columns later, even if the collection is reloaded or
        # mutated in between, so it reads this snapshot instead of re-deriving
        # document order from the live collection.
        self._document_ids = [document.id for document in self.collection.docs]

        for query_id, query in query_items:
            freq_termsets = self.mine_termsets(query, min_freq, stopwords)

            query_vector = self.calculate_ts_idf(freq_termsets)
            weights = self._model_func(freq_termsets)
            # document_vectors is local on purpose: it is the large allocation,
            # and nothing needs it once this query has been scored.
            document_vectors = self.calculate_tsf(freq_termsets)
            weighted_vectors = self._vectorizer(
                document_vectors, query_vector, weights,
            )
            if weighted_vectors is None:
                raise NotImplementedError("A concrete model must implement _vectorizer()")
            scores = cosine_similarity(
                query_vector, weighted_vectors, len(self._document_ids),
            )

            self._query_ids.append(query_id)
            self._queryVectors.append(query_vector)
            self._weights.append(weights)
            self._docScores.append(scores)
            self._termset_counts.append(len(freq_termsets))

        return self


    def evaluate(self, k: int | None = None):
        """Rank the scores fit() produced and return the precision/recall arrays.

        Scoring happens in fit(); this turns each query's stored score vector
        into a ranking and its metrics, so it can be called repeatedly at
        different cutoffs without re-scoring.

        Judgments are selected by fitted query ID. Rankings retain original
        document IDs, and repeated calls replace previous evaluation results.
        Per-query reciprocal ranks are stored in reciprocal_ranks. k is an
        ordinary top-k cutoff (only the first k ranked documents are
        considered); None considers the full ranking. The precision/recall
        values themselves are means at each relevant hit's rank, not a
        single value at rank k -- see evaluation.metrics.calc_precision_recall.
        """
        if k is not None and (
            isinstance(k, bool) or not isinstance(k, int) or k < 1
        ):
            raise ValueError("k must be a positive integer or None")

        number_of_queries = len(self._queryVectors)
        if any(
            len(values) != number_of_queries
            for values in (self._query_ids, self._docScores, self._weights)
        ):
            raise ValueError("Fitted query IDs, vectors, and weights need matching lengths")
        if any(query_id is None for query_id in self._query_ids):
            raise ValueError("Pass a query-ID mapping when fitting a custom query subset")

        for values in (self.precision, self.recall, self.reciprocal_ranks, self.ranking):
            values.clear()

        relevant = self.collection.relevant
        # Use the document order captured at fit() time, not the live
        # collection: scores were computed against that snapshot, and the
        # collection may have been reloaded or mutated since.
        document_ids = self._document_ids
        fitted_queries = zip(self._query_ids, self._docScores, strict=True)
        for query_id, scores in fitted_queries:
            ranking = list(rank_scores(scores, document_ids))
            cutoff = len(ranking) if k is None else k
            precision, recall, reciprocal_rank = calc_precision_recall(
                [str(document_id) for document_id in ranking],
                relevant.get(query_id, ()),
                cutoff,
            )
            self.ranking.append(ranking)
            self.precision.append(round(precision, 8))
            self.recall.append(round(recall, 8))
            self.reciprocal_ranks.append(round(reciprocal_rank, 8))

        return np.array(self.precision), np.array(self.recall)
