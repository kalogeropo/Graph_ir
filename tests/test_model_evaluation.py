"""Verify fitted query identity, original document IDs, and evaluation metrics."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from graph_ir.data.preprocessing.collection import Collection
from graph_ir.infra.evaluation.metrics import calc_precision_recall, evaluate_sim
from graph_ir.infra.models.model import Model


class FixtureModel(Model):
    """Use plain IDF weighting to exercise the shared evaluation pipeline."""

    def get_model(self):
        return self.__class__.__name__

    def _model_func(self, termsets):
        return np.ones(len(termsets))

    def _vectorizer(self, termset_freq_ij, idf, *args):
        return termset_freq_ij * idf.reshape(-1, 1)


class EvaluationHelperTest(unittest.TestCase):
    def test_cosine_ranking_preserves_original_ids(self):
        scores = evaluate_sim(
            [1, 0], [[1, 0, 1], [0, 1, 1]], ["42", "10", "100"]
        )
        self.assertEqual(list(scores), ["42", "100", "10"])
        np.testing.assert_allclose(list(scores.values()), [1, 1 / np.sqrt(2), 0])

    def test_zero_vectors_and_ties_keep_document_order(self):
        self.assertEqual(
            evaluate_sim([0], [[2, 1, 0]], ["42", "10", "100"]),
            {"42": 0.0, "10": 0.0, "100": 0.0},
        )
        self.assertEqual(
            list(evaluate_sim([1], [[2, 1, 0]], ["42", "10", "100"])),
            ["42", "10", "100"],
        )
        self.assertEqual(
            evaluate_sim([], np.zeros((0, 3)), ["42", "10", "100"]),
            {"42": 0.0, "10": 0.0, "100": 0.0},
        )

    def test_invalid_matrix_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_sim([1, 0], [[1, 0]], ["42", "10"])
        with self.assertRaises(ValueError):
            evaluate_sim([1], [[1, 0]], ["42"])

    def test_metric_definitions_and_cutoff(self):
        ranking = ["10", "42", "100"]
        relevant = ["42", "100"]
        np.testing.assert_allclose(
            calc_precision_recall(ranking, relevant, 4), [7 / 12, 3 / 4, 1 / 2]
        )
        self.assertEqual(
            calc_precision_recall(ranking, relevant, 3), (0.5, 0.5, 0.5)
        )
        np.testing.assert_allclose(
            calc_precision_recall(ranking, relevant, 1), [7 / 12, 3 / 4, 1 / 2]
        )
        self.assertEqual(calc_precision_recall(ranking, [], 3), (0, 0, 0))
        self.assertEqual(calc_precision_recall([], relevant, 0), (0, 0, 0))


class ModelEvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        records = [
            {"_id": "42", "tokens": ["a"]},
            {"_id": "10", "tokens": ["b"]},
            {"_id": "100", "tokens": ["a", "b"]},
        ]
        (path / "corpus.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        self.collection = Collection(path).create()
        self.collection.queries = {"q42": "a", "q10": "b", "q100": "a b"}
        self.collection.qrels = {
            "q42": {"42": 1}, "q10": {"10": 1}, "q100": {"100": 1},
        }
        self.model = FixtureModel(self.collection)

    def test_subset_query_order_uses_matching_judgments_and_original_ids(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit(queries={"q100": "a b", "q42": "a"})
            precision, recall = self.model.evaluate()
        self.assertEqual(self.model._query_ids, ["q100", "q42"])
        self.assertEqual(self.model.ranking, [
            ["100", "42", "10"], ["42", "100", "10"],
        ])
        np.testing.assert_array_equal(precision, [1, 1])
        np.testing.assert_array_equal(recall, [1, 1])
        self.assertEqual(self.model.reciprocal_ranks, [1, 1])

    def test_all_queries_are_fitted_and_repeated_calls_replace_state(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit()
            first_precision, first_recall = self.model.evaluate()
            second_precision, second_recall = self.model.evaluate()
        self.assertEqual(len(self.model._query_ids), 3)
        self.assertEqual(len(self.model.ranking), 3)
        self.assertEqual(len(self.model.reciprocal_ranks), 3)
        np.testing.assert_array_equal(first_precision, second_precision)
        np.testing.assert_array_equal(first_recall, second_recall)

        with redirect_stdout(io.StringIO()):
            self.model.fit(queries={"q10": "b"})
            precision, recall = self.model.evaluate()
        self.assertEqual(self.model._query_ids, ["q10"])
        self.assertEqual(self.model.ranking, [["10", "100", "42"]])
        np.testing.assert_array_equal(precision, [1])
        np.testing.assert_array_equal(recall, [1])

    def test_evaluation_uses_current_grades_and_handles_unjudged_queries(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit(queries={"q42": "a", "unjudged": "b"})
            self.collection.qrels["q42"]["42"] = 0
            precision, recall = self.model.evaluate()
        np.testing.assert_array_equal(precision, [0, 0])
        np.testing.assert_array_equal(recall, [0, 0])
        self.assertEqual(self.model.reciprocal_ranks, [0, 0])

    def test_tokenized_query_lists_follow_collection_query_order(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit(queries=[["a"], ["b"], ["a", "b"]])
            precision, recall = self.model.evaluate()
        self.assertEqual(self.model._query_ids, ["q42", "q10", "q100"])
        np.testing.assert_array_equal(precision, [1, 1, 1])
        np.testing.assert_array_equal(recall, [1, 1, 1])

    def test_ambiguous_query_lists_require_ids_for_evaluation(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit(queries=[["a"]])
        with self.assertRaisesRegex(ValueError, "query-ID"):
            self.model.evaluate()

    def test_incomplete_vector_state_is_rejected(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit()
        self.model._weights.pop()
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            self.model.evaluate()

    def test_placeholder_vectorizer_requires_a_concrete_model(self):
        class PlaceholderModel(FixtureModel):
            def _vectorizer(self, termset_freq_ij, idf, *args):
                return None

        model = PlaceholderModel(self.collection)
        with redirect_stdout(io.StringIO()):
            model.fit(queries={"q42": "a"})
        with self.assertRaisesRegex(NotImplementedError, "_vectorizer"):
            model.evaluate()

    def test_empty_query_batch_returns_empty_results(self):
        with redirect_stdout(io.StringIO()):
            self.model.fit(queries={})
            precision, recall = self.model.evaluate()
        self.assertEqual(precision.size, 0)
        self.assertEqual(recall.size, 0)
        self.assertEqual(self.model.ranking, [])

    def test_invalid_cutoffs_are_rejected(self):
        for cutoff in (0, -1, 1.5, True):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises(ValueError):
                    self.model.evaluate(k=cutoff)


if __name__ == "__main__":
    unittest.main()
