# Collections

Tracked input data for Graph_ir. These are separate from the Python code in
`src/graph_ir/data/` and generated outputs in `artifacts/`.

## Selected sources

| Collection | Documents | Queries | Judgments | Status |
| --- | ---: | ---: | --- | --- |
| `CF` | 1,239 | 100 | 100 rows of relevant document IDs | Selected spectral-project snapshot |
| `baeza` | 4 | None supplied | None supplied | Small example, not an evaluation benchmark |
| `test` | 4 | None supplied | None supplied | Small example, not an evaluation benchmark |

Copied without changing file contents from
`to_merge/Spectral-Clustering-and-Query-Expansion-using-Embeddings-on-the-GSB/collections/`.
`CF/raw/` additionally preserves that project's `datasets/CF_RAW/`, including six
original document files and `cfquery` with the original query IDs and assessment
codes. The imported data does not require access to `to_merge/`.

`provenance.json` records the selection and checks. `checksums.sha256` records the
SHA-256 hash of every imported data file. Git text conversion is disabled for this
directory to preserve source bytes. Do not edit these input snapshots in place.

## Format and validation

- CF documents use five-digit filenames (`00001` through `01239`) and one token
  per line. They are already lowercase and punctuation-cleaned; stopwords remain.
  This is a legacy processed snapshot, not raw prose.
- `Queries.txt` has one query per line. Its one-based line number is the query ID.
  Queries retain mixed case and have not had stopwords removed in the stored file.
- `Relevant.txt` has one row per corresponding query, containing whitespace-
  separated document IDs. This is a binary relevance view. Original assessment
  codes remain available in `CF/raw/cfquery`.
- Validation found 1,239 unique consecutive document IDs, no empty CF documents,
  100 aligned query/relevance rows, no duplicate judgments within rows, and no
  judgments referencing absent documents. All 100 relevance sets match the raw
  `cfquery` records, whose declared judgment counts also match.
- This establishes local completeness and internal consistency, not independent
  certification of the dataset or exact reproduction of a published experiment.

## Why these copies

- The GSB and GIRTE CF copies contain only 1,209 documents. Their document files
  match the selected copy where present, and their query/relevance files match
  exactly, but 29 missing document IDs are referenced by 166 judgments.
- Spectral `datasets/parsed_CF/punc_docs` is byte-identical to the selected CF
  documents; it is not duplicated here.
- Spectral `datasets/parsed_CF/docs` differs in 454 documents. For example, file
  `00012` contains only seven title tokens, whereas the selected version has 108
  tokens including the raw record's `EX` text. It is not selected as canonical.
- `punc_stop_docs` and `punc_stop_stem_docs` remain legacy comparison candidates.
  Their generation recipes must be established before treating their names as
  specifications. They are not imported as additional canonical collections.
- The inspected GSB Cranfield copy has 1,400 documents and 225 query/relevance
  rows, but documents `00471` and `00995` are empty. Import is pending investigation.
- The inspected GSB NPL copy has only 1,430 documents, IDs 10000 through 11429,
  and 93 query/relevance rows. It lacks 1,523 distinct relevant document IDs
  referenced by 1,839 judgments. It is not imported as a complete benchmark.

## Preprocessing proposal

Preserve two explicitly different paths:

1. **Reproduce the spectral input.** Load `CF/docs` as existing ordered tokens,
   without re-cleaning, stemming, or removing document stopwords. The spectral
   query loader defaults to Gensim `simple_preprocess(min_len=1, max_len=30)`
   followed by NLTK English stopword removal. Its `prep=False` path instead splits
   the stored query on whitespace. Preserve both settings under named profiles;
   record the selected setting in every experiment.
2. **Build new pipelines from raw text.** Parse record fields and IDs first, keeping
   title (`TI`), abstract (`AB`), and extract (`EX`) distinct. Then apply explicit
   document and query transformations in `graph_ir.data.preprocessing`. Preserve
   sentence boundaries for sentence/window models and original text for BERT.
   Write derived datasets to `artifacts/`, keyed by input hashes and settings.

Do not reuse the old notebook parser unchanged: its `('AB' or 'EX')` expression
only matches `AB`, and its record-flush logic needs validation. Define the text
field selection before claiming that a newly generated corpus matches the legacy
snapshot. For transformer models, decide truncation versus chunking explicitly.

The initial implementation should load and validate this snapshot, then add a
spectral-compatible profile with token-level comparisons. Other projects' profiles
follow independently; sharing a repository does not require identical preprocessing.
No preprocessing implementation or transformation is performed by this import.
