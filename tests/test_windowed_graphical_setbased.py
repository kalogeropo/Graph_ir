"""Compare windowed matrices with hand calculations and the original formula."""

from collections import Counter
from pathlib import Path
from random import Random
import unittest

import networkx as nx
import numpy as np

from graph_ir.data.collection import Collection
from graph_ir.data.document import Document
from graph_ir.models.graphical_setbased import GSBModel
from graph_ir.models.windowed_graphical_setbased import WindowedGSBModel

BAEZA_PATH = Path(__file__).resolve().parents[1] / "collections" / "baeza"


def collection_from_tokens(tokens):
    collection = Collection(BAEZA_PATH, docs=[Document(tokens, id="42")])
    collection.inverted_index = collection.create_inverted_index()
    collection.doc_id_to_position = {"42": 0}
    return collection


def original_matrix(document, window, minimum_seven, terms=None):
    """Reproduce the supplied full-vocabulary loops for valid original sizes."""
    if terms is None:
        terms = list(Counter(document.terms))
    size = window if isinstance(window, int) else int(window * len(document.terms))
    if minimum_seven:
        size = max(7, size)
    matrix = np.zeros((len(terms), len(terms)), dtype=int)
    for start in range(0, len(document.terms), size):
        counts = Counter(document.terms[start:start + size])
        for i, left in enumerate(terms):
            for j, right in enumerate(terms):
                if left in counts and right in counts:
                    if i == j:
                        matrix[i, j] += counts[left] * (counts[left] + 1) // 2
                    else:
                        matrix[i, j] += counts[left] * counts[right]
    return matrix


class WindowedGraphicalSetbasedTest(unittest.TestCase):
    def test_hand_calculated_counts_include_final_partial_window(self):
        collection = collection_from_tokens(["a", "b", "a", "a", "b"])
        model = WindowedGSBModel(collection, window=2, window_cut_off=False)
        np.testing.assert_array_equal(model.doc_to_matrix(collection.docs[0]), [[4, 1], [1, 2]])

    def test_term_order_and_subsets_match_the_parent_interface(self):
        collection = collection_from_tokens(["a", "b", "a", "a", "b"])
        model = WindowedGSBModel(collection, window=2, window_cut_off=False)
        document = collection.docs[0]
        np.testing.assert_array_equal(model.doc_to_matrix(document, ["b", "a"]), [[2, 1], [1, 4]])
        np.testing.assert_array_equal(model.doc_to_matrix(document, ["b"]), [[2]])
        self.assertEqual(model.doc_to_matrix(document, []).shape, (0, 0))

    def test_cutoff_enforces_seven_tokens_and_keeps_the_tail(self):
        collection = collection_from_tokens(list("abcdefgh"))
        model = WindowedGSBModel(collection, window=2, window_cut_off=True)
        expected = np.zeros((8, 8), dtype=int)
        expected[:7, :7] = 1
        expected[7, 7] = 1
        np.testing.assert_array_equal(model.doc_to_matrix(collection.docs[0]), expected)

    def test_fractional_window_rounds_down_and_handles_short_documents(self):
        collection = collection_from_tokens(list("abcdefghij"))
        model = WindowedGSBModel(collection, window=.29, window_cut_off=False)
        np.testing.assert_array_equal(
            model.doc_to_matrix(collection.docs[0]),
            original_matrix(collection.docs[0], 2, False),
        )
        short = collection_from_tokens(["a", "a", "b"])
        singleton = WindowedGSBModel(short, window=.1, window_cut_off=False)
        np.testing.assert_array_equal(singleton.doc_to_matrix(short.docs[0]), [[2, 0], [0, 1]])

    def test_empty_and_single_window_documents(self):
        for window in (1, 8, .1, 1.0):
            with self.subTest(window=window):
                empty = collection_from_tokens([])
                model = WindowedGSBModel(empty, window=window, window_cut_off=False)
                self.assertEqual(model.doc_to_matrix(empty.docs[0]).shape, (0, 0))
                self.assertEqual(len(model.graph), 0)
        collection = collection_from_tokens(["a", "a", "b"])
        expected = GSBModel(collection).doc_to_matrix(collection.docs[0])
        for window in (3, 20, 1.0, 2.0):
            with self.subTest(window=window):
                model = WindowedGSBModel(collection, window=window, window_cut_off=False)
                np.testing.assert_array_equal(model.doc_to_matrix(collection.docs[0]), expected)

    def test_baeza_and_random_documents_match_the_original_formula(self):
        random = Random(9237)
        collections = [Collection(BAEZA_PATH).create()]
        collections.extend(
            collection_from_tokens([random.choice("abcdefgh") for _ in range(random.randint(10, 60))])
            for _ in range(30)
        )
        for collection in collections:
            for window in (1, 2, 7, 8, 100, .1, .5, 1.0, 2.0):
                for cutoff in (False, True):
                    model = WindowedGSBModel(collection, window=window, window_cut_off=cutoff)
                    for document in collection.docs:
                        with self.subTest(document=document.terms, window=window, cutoff=cutoff):
                            actual = model.doc_to_matrix(document)
                            np.testing.assert_array_equal(actual, original_matrix(document, window, cutoff))
                            self.assertTrue(np.issubdtype(actual.dtype, np.integer))

    def test_union_graph_uses_the_windowed_matrix_for_core_weighting(self):
        collection = collection_from_tokens(["a", "b", "c", "c", "d", "d"])
        model = WindowedGSBModel(
            collection, window=3, window_cut_off=False, k_core_bool=True, h_val=2,
        )
        self.assertEqual(nx.get_node_attributes(model.graph, "weight"), {"a": 2, "b": 2, "c": 4, "d": 3})
        self.assertEqual(model.graph["a"]["b"]["weight"], 2)
        self.assertEqual(model.graph["c"]["d"]["weight"], 2)
        self.assertFalse(model.graph.has_edge("a", "d"))
        self.assertEqual(nx.number_of_selfloops(model.graph), 0)
        collection.queries = {"q1": "a b"}
        collection.qrels = {"q1": {"42": 1}}
        model.fit().evaluate()
        self.assertEqual(model._query_ids, ["q1"])
        self.assertEqual(model.ranking, [["42"]])

    def test_invalid_window_sizes_are_rejected_before_building_graphs(self):
        collection = collection_from_tokens([])
        for window in (0, -1, 0.0, -.1, float("nan"), float("inf"), True, "8", None):
            with self.subTest(window=window), self.assertRaisesRegex(ValueError, "window must"):
                WindowedGSBModel(collection, window=window)


if __name__ == "__main__":
    unittest.main()
