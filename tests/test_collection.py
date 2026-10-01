"""Tests for collection loading, indexing, and relevance views."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from graph_ir.data.collection import Collection


BAEZA_PATH = Path(__file__).resolve().parents[1] / "collections" / "baeza"


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.collection = Collection(BAEZA_PATH).create().load_queries().load_qrels()

    def test_loads_baeza_documents_and_preserves_tokens(self):
        self.assertEqual(self.collection.num_docs, 4)
        self.assertEqual([doc.id for doc in self.collection.docs], ["1", "2", "3", "4"])
        self.assertEqual(
            self.collection.docs[0].terms,
            ["a", "b", "c", "a", "d", "a", "d", "c", "a", "b"],
        )

    def test_index_contains_correct_frequencies_and_document_ids(self):
        index = self.collection.inverted_index
        self.assertEqual(index["a"]["total_tf"], 6)
        self.assertEqual(index["a"]["posting_list"], {"1": 4, "2": 2})
        self.assertEqual(index["l"]["posting_list"], {"4": 3})

    def test_limits_and_repeated_loading_rebuild_the_index(self):
        self.collection.create(first=2)
        self.assertEqual(self.collection.num_docs, 2)
        self.assertNotIn("l", self.collection.inverted_index)
        self.collection.create()
        self.assertEqual(self.collection.num_docs, 4)
        self.assertEqual(self.collection.inverted_index["a"]["total_tf"], 6)
        self.collection.create(first=0)
        self.assertEqual(self.collection.docs, [])
        self.assertEqual(self.collection.inverted_index, {})
        with self.assertRaises(ValueError):
            self.collection.create(first=-1)

    def test_queries_keep_their_explicit_ids_and_text(self):
        self.assertEqual(
            self.collection.queries,
            {
                "1": "this the a test",
                "2": "this the b test",
                "3": "this the c test",
                "4": "this the d test",
            },
        )

    def test_duplicate_query_id_is_rejected_without_overwriting_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            collection = Collection(directory)
            collection.queries = {"existing": "text"}
            (Path(directory) / "queries.jsonl").write_text(
                json.dumps({"_id": "1", "text": "a"}) + "\n"
                + json.dumps({"_id": "1", "text": "b"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate query ID: 1"):
                collection.load_queries()
            self.assertEqual(collection.queries, {"existing": "text"})

    def test_qrels_preserve_zero_scores_and_missing_document_judgments(self):
        self.assertEqual(
            self.collection.qrels["1"], {"1": 1, "2": 1, "3": 0, "4": 0}
        )
        self.assertEqual(sum(len(v) for v in self.collection.qrels.values()), 16)
        self.collection.create(first=2).load_qrels()
        self.assertEqual(self.collection.qrels["1"]["4"], 0)

    def test_relevant_property_tracks_qrels_without_changing_them(self):
        self.assertEqual(self.collection.relevant["1"], ["1", "2"])
        returned_view = self.collection.relevant
        returned_view["1"].clear()
        self.assertEqual(self.collection.relevant["1"], ["1", "2"])

        self.collection.qrels["1"]["3"] = 4
        self.assertEqual(self.collection.relevant["1"], ["1", "2", "3"])
        self.assertEqual(self.collection.qrels["1"]["3"], 4)
        self.collection.queries["5"] = "unjudged query"
        self.collection.qrels["6"] = {"1": 0}
        self.assertEqual(self.collection.relevant["5"], [])
        self.assertEqual(self.collection.relevant["6"], [])

    def test_graded_qrels_and_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            qrels_path = path / "qrels.tsv"
            qrels_path.write_text(
                "query-id\tcorpus-id\tscore\n10\t99\t0\n10\t42\t2\n10\t77\t4\n",
                encoding="utf-8",
            )
            collection = Collection(path).load_qrels()
            self.assertEqual(collection.qrels["10"], {"99": 0, "42": 2, "77": 4})
            self.assertEqual(collection.relevant["10"], ["42", "77"])

            qrels_path.write_text("query-id\tcorpus-id\tscore\n", encoding="utf-8")
            collection.load_qrels()
            self.assertEqual(collection.qrels, {})

    def test_malformed_qrels_are_rejected_without_overwriting_existing_data(self):
        invalid_files = [
            "query-id\tcorpus-id\n1\t1\n",
            "query-id\tcorpus-id\tscore\n1\t\t1\n",
            "query-id\tcorpus-id\tscore\n1\t1\tinvalid\n",
            "query-id\tcorpus-id\tscore\n1\t1\t1\n1\t1\t2\n",
        ]
        with tempfile.TemporaryDirectory() as directory:
            collection = Collection(directory)
            collection.qrels = {"existing": {"doc": 2}}
            for content in invalid_files:
                with self.subTest(content=content):
                    (Path(directory) / "qrels.tsv").write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        collection.load_qrels()
                    self.assertEqual(collection.qrels, {"existing": {"doc": 2}})

    def test_raw_fields_nonconsecutive_ids_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            records = [
                {"_id": "10", "title": "Title", "abstract": "Body", "extract": "Other"},
                {"_id": "42", "tokens": ["Keep", "Keep", "case!"]},
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            collection = Collection(directory).create(fields=("title", "abstract"))
            self.assertEqual(collection.docs[0].terms, ["Title", "Body"])
            self.assertEqual(collection.docs[1].terms, ["Keep", "Keep", "case!"])
            self.assertEqual(
                collection.inverted_index["Keep"]["posting_list"], {"42": 2}
            )

            path.write_text(
                json.dumps(records[0]) + "\n" + json.dumps(records[0]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate document ID"):
                collection.create()


class StopwordsLazyLoadingTest(unittest.TestCase):
    """Constructing/loading a Collection must never require nltk data;
    only actually reading .stopwords should touch it, and only once."""

    def test_construction_does_not_load_stopwords(self):
        with patch("graph_ir.data.collection.stopwords.words") as mocked_words:
            Collection(BAEZA_PATH)
            mocked_words.assert_not_called()

    def test_create_and_query_loading_do_not_load_stopwords(self):
        with patch("graph_ir.data.collection.stopwords.words") as mocked_words:
            Collection(BAEZA_PATH).create().load_queries().load_qrels()
            mocked_words.assert_not_called()

    def test_first_read_loads_stopwords_once_and_caches_the_result(self):
        with patch(
            "graph_ir.data.collection.stopwords.words", return_value=["a", "the"]
        ) as mocked_words:
            collection = Collection(BAEZA_PATH)
            mocked_words.assert_not_called()

            first = collection.stopwords
            second = collection.stopwords

            mocked_words.assert_called_once_with("english")
            self.assertEqual(first, ["a", "the"])
            self.assertIs(first, second)

    def test_assigning_stopwords_directly_bypasses_nltk_entirely(self):
        with patch("graph_ir.data.collection.stopwords.words") as mocked_words:
            collection = Collection(BAEZA_PATH)
            collection.stopwords = {"and"}
            self.assertEqual(collection.stopwords, {"and"})
            mocked_words.assert_not_called()

    def test_missing_corpus_is_downloaded_once_on_first_read_not_silently(self):
        with (
            patch(
                "graph_ir.data.collection.stopwords.words",
                side_effect=[LookupError("missing"), ["a", "the"]],
            ) as mocked_words,
            patch("graph_ir.data.collection.nltk.download") as mocked_download,
            patch("builtins.print") as mocked_print,
        ):
            collection = Collection(BAEZA_PATH)
            mocked_download.assert_not_called()  # never at construction

            result = collection.stopwords

            mocked_download.assert_called_once_with("stopwords", quiet=True)
            self.assertEqual(mocked_words.call_count, 2)  # retried after downloading
            self.assertEqual(result, ["a", "the"])
            mocked_print.assert_called_once()  # not silent


if __name__ == "__main__":
    unittest.main()
