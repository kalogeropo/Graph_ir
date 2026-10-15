from collections.abc import Callable
import csv
import json
from pathlib import Path

import nltk
from nltk.corpus import stopwords

from graph_ir.data.document import Document
from graph_ir.indexing.inverted_index import build_inverted_index, export_inverted_index


class Collection:
    def __init__(self, path: str | Path , docs=None):

        self._stopwords = None
        self.path = Path(path)
        self.docs = list(docs) if docs is not None else []

        self.queries = {}
        self.qrels = {}
        self.inverted_index = {}
        self.doc_id_to_position = {}
        self.preprocessor: Callable[[str | list[str]], list[str]] | None = None

    @property
    def num_docs(self) -> int:
        """Number of loaded documents.

        Derived rather than stored: it is the column count every termset matrix
        is built against, so a stored copy left stale by a direct mutation of
        docs would silently disagree with the document order models snapshot.
        """
        return len(self.docs)

    @property
    def stopwords(self):
        """English stopwords.

        Loaded from nltk on first use, not at construction, so merely
        building a Collection never requires the nltk 'stopwords' corpus.
        If the corpus is missing when first actually needed, it is
        downloaded then (never at construction, and never silently -- a
        message is printed first). Assign a value directly to override it.
        """
        if self._stopwords is None:
            try:
                self._stopwords = stopwords.words("english")
            except LookupError:
                print("nltk 'stopwords' corpus not found; downloading it now...")
                nltk.download("stopwords", quiet=True)
                self._stopwords = stopwords.words("english")
        return self._stopwords

    @stopwords.setter
    def stopwords(self, value):
        self._stopwords = value

    def create(self, first: int | None = None,
               fields=("title", "text", "abstract", "extract"), *,
               preprocessor: Callable[[str | list[str]], list[str]] | None = None):
        """Load documents, optionally preprocess their terms, and build the index.

        The preprocessor receives raw text or stored tokens and returns a token
        list. Original document text and IDs are preserved. Models use this same
        function for queries during fitting; loaded query text stays unchanged.
        Omitting it preserves whitespace tokenization and any existing tokens.
        Each call reloads the source documents before applying the chosen rules.
        """
        if first is not None and first < 0:
            raise ValueError("first must be nonnegative or None")

        documents = []
        seen_ids = set()

        with (self.path / "corpus.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if first is not None and len(documents) >= first:
                    break

                record = json.loads(line)
                document_id = record["_id"]

                if document_id in seen_ids:
                    raise ValueError(f"Duplicate document ID: {document_id}")

                if "tokens" in record:
                    text = record["tokens"]
                else:
                    text = "\n".join(
                        record[field] for field in fields
                        if record.get(field)
                    )

                document = Document(text=text, id=document_id)
                if preprocessor is not None:
                    document.terms = preprocessor(text)
                    document.num_of_words = len(document.terms)
                documents.append(document)
                seen_ids.add(document_id)

        self.docs = documents
        self.inverted_index = self.create_inverted_index()
        self.doc_id_to_position = {
            document.id: position
            for position, document in enumerate(self.docs)
        }
        self.preprocessor = preprocessor

        return self

    def create_inverted_index(self):
        """Build an index whose postings map document IDs to term frequencies."""
        return build_inverted_index(self.docs)

    def export_inverted_index(self, path: str | Path | None = None) -> list[dict]:
        """Flatten this collection's inverted index into JSON-serializable records.

        See indexing.inverted_index.export_inverted_index for the record shape.
        Pass a path under artifacts/ to also write the records as JSON there.
        """
        return export_inverted_index(self.inverted_index, path)

    def load_queries(self) -> "Collection":
        """Load query text keyed by explicit query ID.

        Raises:
            ValueError: If a query ID occurs more than once.
        """
        qs = {}
        with (self.path / "queries.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                query = json.loads(line)
                query_id = str(query["_id"])
                if query_id in qs:
                    raise ValueError(f"Duplicate query ID: {query_id}")
                qs[query_id] = query["text"]
        self.queries = qs
        return self


    @property
    def relevant(self) -> dict[str, set[str]]:
        """Return relevant document IDs per query for binary evaluation.

        Scores of at least 1 count as relevant. Known queries without judgments
        have empty sets. Graded qrels remain unchanged for nDCG.

        Sets, not lists: these are only ever membership-tested, once per ranked
        document, and CF has up to 304 relevant documents for a single query --
        scanning that as a list made scoring 100 queries about 26x slower for
        identical results.
        """
        relevant = {
            query_id: {
                document_id
                for document_id, score in judgments.items()
                if score >= 1
            }
            for query_id, judgments in self.qrels.items()
        }
        for query_id in self.queries:
            relevant.setdefault(query_id, set())
        return relevant

    def load_qrels(self, filename: str = "qrels.tsv") -> "Collection":
        """Load relevance scores keyed by query ID and document ID.

        Args:
            filename: TSV filename relative to the collection directory.

        Scores, including zero, are preserved. IDs remain strings, and judgments
        for documents outside the loaded collection are retained.

        Raises:
            ValueError: If required columns, IDs, or integer scores are invalid,
                or a query/document pair occurs more than once.
        """
        qrels = {}

        with (self.path / filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            required = {"query-id", "corpus-id", "score"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Qrels must contain query-id, corpus-id, and score columns")

            for row in reader:
                query_id = row["query-id"]
                document_id = row["corpus-id"]
                if not query_id or not document_id:
                    raise ValueError(f"Missing ID at qrels line {reader.line_num}")

                try:
                    score = int(row["score"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid integer score at qrels line {reader.line_num}"
                    ) from error

                judgments = qrels.setdefault(query_id, {})
                if document_id in judgments:
                    raise ValueError(
                        f"Duplicate judgment for query {query_id}, document {document_id}"
                    )
                judgments[document_id] = score

        self.qrels = qrels
        return self
