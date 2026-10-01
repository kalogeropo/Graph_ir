from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy import array

from graph_ir.data.collection import Collection
from graph_ir.data.preprocessing.text import remove_stopwords
from graph_ir.models.apriori import apriori
from graph_ir.evaluation.metrics import calc_precision_recall, evaluate_sim


class Model(ABC):
    def __init__(self,collection: Collection):
        self.collection = collection
        self._model = "base abstract model"

        self._queryVectors = []
        self._docVectors = []
        self._weights = []
        self._query_ids = []
        self._document_ids = []
        # metrics
        self.precision = []
        self.recall = []
        self.reciprocal_ranks = []
        # model_document_ranking
        self.ranking = []

    @abstractmethod
    def get_model(self):
        return self.__class__.__name__
    @abstractmethod
    def _model_func(self, freq_termsets: Any) -> np.ndarray:
        pass

    # the base model fit function will implement the set based document Queries representation. can be overriden
    # in any subclass at will.
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
        pass

    def calculate_tsf(self, termsets: dict) -> np.ndarray:
        """Build a termset-by-document matrix using direct frequency lookups.

        termsets maps each nonempty termset to the IDs of documents containing all
        its terms, as produced by Apriori. Rows follow termset insertion order;
        columns follow collection.docs order. Frequencies use logarithmic weighting
        and rounding: round(1 + log2(minimum term frequency), 3).
        """
        matrix = np.zeros((len(termsets), self.collection.num_docs))
        frequency_weights = {}

        for row, (termset, document_ids) in enumerate(termsets.items()):
            frequencies = [self.collection.inverted_index[term]["posting_list"] for term in termset]
            for document_id in document_ids:
                frequency = min(lookup[document_id] for lookup in frequencies)
                column = self.collection.doc_id_to_position[document_id]
                weight = frequency_weights.get(frequency)
                if weight is None:
                    weight = round(1 + np.log2(frequency), 3)
                    frequency_weights[frequency] = weight
                matrix[row, column] = weight

        return matrix
    def calculate_ts_idf(self, termsets):
        # len(value) => in how many documents each termset appears
        return array([round(np.log2(1 + (self.collection.num_docs / len(value))), 3) for value in termsets.values()])

    def fit(self, queries=None, min_freq: int = 1, stopwords: bool = False):
        """Build query-dependent vectors while retaining query identities.

        Query mappings carry explicit IDs. Tokenized query lists use collection
        query order when their lengths match; subsets need an ID mapping for
        evaluation. When the collection has a preprocessor, the same rules are
        applied to raw and tokenized queries. Otherwise, raw strings are split
        on whitespace. Optional query stopword removal follows tokenization
        and is case-insensitive against collection.stopwords regardless of
        whether a preprocessor already lowercased the tokens.
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
            self._query_ids, self._queryVectors, self._docVectors, self._weights,
            self.precision, self.recall, self.reciprocal_ranks, self.ranking,
        ):
            values.clear()

        # Snapshot document identity now: calculate_tsf lays out matrix columns
        # using the collection as it is at this moment. evaluate() must label
        # those same columns later, even if the collection is reloaded or
        # mutated in between, so it reads this snapshot instead of re-deriving
        # document order from the live collection.
        self._document_ids = [document.id for document in self.collection.docs]

        for i, (query_id, query) in enumerate(query_items, start=1):
            if self.collection.preprocessor is not None:
                query = self.collection.preprocessor(query)
            elif isinstance(query, str):
                query = query.split()
            if stopwords:
                query = remove_stopwords(
                    query, self.collection.stopwords, case_insensitive=True,
                )

            freq_termsets = apriori(query, self.collection.inverted_index, min_freq)

            query_vector = self.calculate_ts_idf(freq_termsets)
            document_vectors = self.calculate_tsf(freq_termsets)
            weights = self._model_func(freq_termsets)

            self._query_ids.append(query_id)
            self._queryVectors.append(query_vector)
            self._docVectors.append(document_vectors)
            self._weights.append(weights)

        return self


    def evaluate(self, k: int | None = None):
        """Rank fitted queries and return the precision/recall arrays.

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
            for values in (self._query_ids, self._docVectors, self._weights)
        ):
            raise ValueError("Fitted query IDs, vectors, and weights need matching lengths")
        if any(query_id is None for query_id in self._query_ids):
            raise ValueError("Pass a query-ID mapping when fitting a custom query subset")

        for values in (self.precision, self.recall, self.reciprocal_ranks, self.ranking):
            values.clear()

        relevant = self.collection.relevant
        # Use the document order captured at fit() time, not the live
        # collection: matrix columns were laid out against that snapshot, and
        # the collection may have been reloaded or mutated since.
        document_ids = self._document_ids
        fitted_queries = zip(
            self._query_ids, self._queryVectors, self._docVectors,
            self._weights, strict=True,
        )
        for i, (query_id, query_vector, document_vectors, weights) in enumerate(
            fitted_queries, start=1
        ):
            weighted_vectors = self._vectorizer(document_vectors, query_vector, weights)
            if weighted_vectors is None:
                raise NotImplementedError("A concrete model must implement _vectorizer()")
            document_similarities = evaluate_sim(
                query_vector, weighted_vectors, document_ids
            )
            ranking = list(document_similarities)
            cutoff = len(ranking) if k is None else k
            precision, recall, reciprocal_rank = calc_precision_recall(
                [str(document_id) for document_id in ranking],
                relevant.get(query_id, []),
                cutoff,
            )
            self.ranking.append(ranking)
            self.precision.append(round(precision, 8))
            self.recall.append(round(recall, 8))
            self.reciprocal_ranks.append(round(reciprocal_rank, 8))
            # print(
            #     f"=> Query {query_id} ({i}/{number_of_queries}), "
            #     f"precision = {precision:.3f}, recall = {recall:.3f}"
            # )

        return np.array(self.precision), np.array(self.recall)
