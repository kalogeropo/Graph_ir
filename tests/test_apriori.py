"""Check frequent termsets against document contents and model query inputs."""

from contextlib import redirect_stdout
import io
from itertools import combinations
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from graph_ir.data.collection import Collection
from graph_ir.models.model import Model
from graph_ir.models.apriori import apriori, create_candidate_k


BAEZA_PATH = Path(__file__).resolve().parents[1] / "collections" / "baeza"


def exhaustive_termsets(query, documents, min_freq):
    """Count support directly from documents, without Apriori or the index."""
    terms = list(dict.fromkeys(query))
    expected = {}
    for size in range(1, len(terms) + 1):
        for combination in combinations(terms, size):
            termset = frozenset(combination)
            document_ids = {
                document.id
                for document in documents
                if termset.issubset(document.terms)
            }
            if len(document_ids) >= min_freq:
                expected[termset] = document_ids
    return expected


class AprioriTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = Collection(BAEZA_PATH).create()

    def test_baeza_matches_exhaustive_document_support(self):
        query = list(self.collection.inverted_index)
        for min_freq in range(1, 6):
            with self.subTest(min_freq=min_freq):
                self.assertEqual(
                    apriori(query, self.collection.inverted_index, min_freq),
                    exhaustive_termsets(query, self.collection.docs, min_freq),
                )

    def test_repeated_and_unknown_terms_preserve_expected_candidates(self):
        result = apriori(["a", "d", "a", "missing"], self.collection.inverted_index, 2)
        self.assertEqual(result, {
            frozenset({"a"}): {"1", "2"},
            frozenset({"d"}): {"1", "2", "3", "4"},
            frozenset({"a", "d"}): {"1", "2"},
        })

    def test_original_ids_survive_dictionary_postings(self):
        index = {
            "a": {"posting_list": {"1": 2, "10": 1, "100": 1}},
            "d": {"posting_list": {"10": 4, "100": 2}},
        }
        self.assertEqual(apriori(["a", "d"], index, 2), {
            frozenset({"a"}): {"1", "10", "100"},
            frozenset({"d"}): {"10", "100"},
            frozenset({"a", "d"}): {"10", "100"},
        })

    def test_support_counts_documents_instead_of_occurrences(self):
        index = {"a": {"posting_list": {"1": 100, "42": 1}}}
        self.assertEqual(apriori(["a"], index, 2), {
            frozenset({"a"}): {"1", "42"},
        })
        self.assertEqual(apriori(["a"], index, 3), {})

    def test_empty_and_unknown_queries_have_no_candidates(self):
        index = self.collection.inverted_index
        self.assertEqual(apriori([], index, 1), {})
        self.assertEqual(apriori(["missing"], index, 1), {})
        self.assertEqual(apriori(["a"], {}, 1), {})
        self.assertEqual(apriori(["a"], {"a": {"posting_list": {}}}, 1), {})

    def test_invalid_minimum_support_is_rejected(self):
        for min_freq in (0, -1, 1.5, "2", True, None):
            with self.subTest(min_freq=min_freq):
                with self.assertRaises(ValueError):
                    apriori([], {}, min_freq)

    def test_raw_query_text_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "token"):
            apriori("a d", self.collection.inverted_index, 1)

    def test_candidates_with_missing_frequent_subsets_are_pruned(self):
        frequent_pairs = {
            frozenset({"a", "b"}): {"1", "2"},
            frozenset({"a", "c"}): {"2", "3"},
        }
        self.assertEqual(create_candidate_k(frequent_pairs, 1), {})

        frequent_pairs[frozenset({"b", "c"})] = {"2", "4"}
        self.assertEqual(create_candidate_k(frequent_pairs, 1), {
            frozenset({"a", "b", "c"}): {"2"},
        })

    def test_join_groups_preserve_pairwise_order_across_levels(self):
        parents = {
            frozenset(terms): {"shared", str(i)}
            for i, terms in enumerate(("cd", "ac", "bd", "ab", "bc", "ad"))
        }
        triples = create_candidate_k(parents, 1)
        self.assertEqual(list(triples), [
            frozenset("acd"), frozenset("bcd"),
            frozenset("abc"), frozenset("abd"),
        ])
        self.assertTrue(all(documents == {"shared"} for documents in triples.values()))
        self.assertEqual(
            create_candidate_k(triples, 2),
            {frozenset("abcd"): {"shared"}},
        )

    def test_termset_order_follows_legacy_levels_and_query_order(self):
        result = apriori(["d", "a", "d", "b"], self.collection.inverted_index, 1)
        self.assertEqual(list(result), [
            frozenset({"d"}), frozenset({"a"}), frozenset({"b"}),
            frozenset({"d", "a"}), frozenset({"d", "b"}),
            frozenset({"a", "b"}), frozenset({"d", "a", "b"}),
        ])


class QueryInputModel(Model):
    """Supply abstract hooks for testing the shared query-input handling."""

    def get_model(self):
        return self.__class__.__name__

    def _model_func(self, termsets):
        return np.ones(len(termsets))

    def _vectorizer(self, termset_freq_ij, idf, *args):
        return termset_freq_ij * idf.reshape(-1, 1)


class ModelQueryInputTest(unittest.TestCase):
    def setUp(self):
        self.collection = Collection(BAEZA_PATH).create()

    def test_fit_tokenizes_query_text_before_stopword_filtering(self):
        self.collection.queries = {"42": "a and d"}
        self.collection.stopwords = {"and"}
        model = QueryInputModel(self.collection)
        with patch(
            "graph_ir.models.model.apriori", wraps=apriori
        ) as miner, redirect_stdout(io.StringIO()):
            self.assertIs(model.fit(min_freq=2, stopwords=True), model)
        self.assertEqual(miner.call_args.args[0], ["a", "d"])
        self.assertEqual(self.collection.queries, {"42": "a and d"})

    def test_fit_accepts_legacy_tokenized_query_lists(self):
        model = QueryInputModel(self.collection)
        with patch(
            "graph_ir.models.model.apriori", wraps=apriori
        ) as miner, redirect_stdout(io.StringIO()):
            model.fit(queries=[["a", "d"]], min_freq=2)
        self.assertEqual(miner.call_args.args[0], ["a", "d"])


if __name__ == "__main__":
    unittest.main()
