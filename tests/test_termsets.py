"""Compare optimized termset weighting with the original posting-list scans."""

from itertools import combinations
import json
from math import log2
from pathlib import Path
import tempfile
import unittest

import numpy as np

from graph_ir.data.collection import Collection
from graph_ir.models.model import Model


BAEZA_PATH = Path(__file__).resolve().parents[1] / "collections" / "baeza"


class FixtureModel(Model):
    """Minimal concrete Model subclass for testing termset functions."""

    def get_model(self):
        return "fixture"

    def _model_func(self, freq_termsets):
        return np.ones(len(freq_termsets))

    def _vectorizer(self, tsf_ij, idf, *args):
        return tsf_ij * idf.reshape(-1, 1)


def legacy_tsf(collection, termsets):
    """Reproduce legacy scans using posting pairs and explicit matrix columns."""
    matrix = np.zeros((len(termsets), collection.num_docs))
    for row, (termset, documents) in enumerate(termsets.items()):
        temporary = {}
        for term in termset:
            posting_pairs = list(collection.inverted_index[term]["posting_list"].items())
            for document_id, frequency in posting_pairs:
                if document_id in documents:
                    temporary.setdefault(document_id, []).append(frequency)
        for document_id, frequencies in temporary.items():
            column = collection.doc_id_to_position[document_id]
            matrix[row, column] = round(1 + log2(min(frequencies)), 3)
    return matrix


class TermsetMatrixTest(unittest.TestCase):
    def setUp(self):
        self.collection = Collection(BAEZA_PATH).create()
        self.model = FixtureModel(self.collection)

    def test_matches_legacy_for_all_baeza_termsets_up_to_size_three(self):
        termsets = {}
        terms = list(self.collection.inverted_index)
        for size in (1, 2, 3):
            for terms_in_set in combinations(terms, size):
                document_ids = set.intersection(*(
                    set(self.collection.inverted_index[term]["posting_list"]) for term in terms_in_set
                ))
                if document_ids:
                    termsets[frozenset(terms_in_set)] = sorted(document_ids)
        np.testing.assert_array_equal(
            self.model.calculate_tsf(termsets),
            legacy_tsf(self.collection, termsets),
        )

    def test_hand_calculated_minimum_frequencies(self):
        termsets = {frozenset({"a"}): ["1", "2"],
                    frozenset({"a", "d"}): ["1", "2"],
                    frozenset({"l"}): ["4"]}
        np.testing.assert_array_equal(
            self.model.calculate_tsf(termsets),
            [[3.0, 2.0, 0.0, 0.0],
             [2.0, 2.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 2.585]],
        )

    def test_nonconsecutive_ids_use_document_order(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [{"_id": "42", "tokens": ["a", "a", "b"]},
                       {"_id": "10", "tokens": ["a", "b", "b"]}]
            (Path(directory) / "corpus.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            collection = Collection(directory).create()
            model = FixtureModel(collection)
            self.assertEqual(collection.doc_id_to_position, {"42": 0, "10": 1})
            np.testing.assert_array_equal(
                model.calculate_tsf({frozenset({"a"}): ["10", "42"]}),
                [[2.0, 1.0]],
            )
            self.assertEqual(collection.docs[0].id, "42")

    def test_repeated_loading_rebuilds_lookup_state(self):
        self.assertEqual(self.collection.inverted_index["a"]["posting_list"], {"1": 4, "2": 2})
        self.collection.create(first=2)
        self.assertEqual(self.collection.doc_id_to_position, {"1": 0, "2": 1})
        self.assertNotIn("l", self.collection.inverted_index)
        self.collection.create(first=0)
        self.assertEqual(self.collection.doc_id_to_position, {})
        self.assertEqual(self.collection.inverted_index, {})
        self.model = FixtureModel(self.collection)
        self.assertEqual(self.model.calculate_tsf({}).shape, (0, 0))

    def test_empty_termsets_preserve_document_dimension(self):
        self.assertEqual(self.model.calculate_tsf({}).shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
