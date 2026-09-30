import csv
import json
from collections import Counter
from pathlib import Path
from nltk.corpus import stopwords

from graph_ir.data.preprocessing.document import Document




class Collection:
    def __init__(self, path: str | Path , docs=None):

        self.stopwords = stopwords.words("english")
        self.path = Path(path)
        self.docs = list(docs) if docs is not None else []
        self.num_docs = len(self.docs)

        self.queries = {}
        self.qrels = {}
        self.inverted_index = {}
        self.doc_id_to_position = {}



    def create(self, first: int | None = None,
               fields=("title", "text", "abstract", "extract")):
        """Load documents and build their inverted index."""
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

                documents.append(Document(text=text, id=document_id))
                seen_ids.add(document_id)

        self.docs = documents
        self.num_docs = len(documents)
        self.inverted_index = self.create_inverted_index()
        self.doc_id_to_position = {
            document.id: position
            for position, document in enumerate(self.docs)
        }

        return self

    def create_inverted_index(self):
        """Build an index whose postings map document IDs to term frequencies."""
        index = {}

        for document in self.docs:
            for term, frequency in Counter(document.terms).items():
                if term not in index:
                    index[term] = {
                        "id": len(index),
                        "term": term,
                        "total_tf": 0,
                        "posting_list": {},
                    }

                index[term]["total_tf"] += frequency
                index[term]["posting_list"][document.id] = frequency

        return index

    def load_queries(self):
        qs = {}
        with (self.path / "queries.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                query = json.loads(line)
                qs[str(query["_id"])] = query["text"]
        self.queries = qs
        return self


    @property
    def relevant(self) -> dict[str, list[str]]:
        """Return relevant document IDs per query for binary evaluation.

        Scores of at least 1 count as relevant. Known queries without judgments
        have empty lists. Graded qrels remain unchanged for nDCG.
        """
        relevant = {
            query_id: [
                document_id
                for document_id, score in judgments.items()
                if score >= 1
            ]
            for query_id, judgments in self.qrels.items()
        }
        for query_id in self.queries:
            relevant.setdefault(query_id, [])
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