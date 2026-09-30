# Graph_ir

One Python package for the complete retrieval workflows from the four projects
under `to_merge/`: parsing, preprocessing, indexing, models, evaluation,
experiments, and analysis.

The repository contains package scaffolding and validated collection snapshots.
No models or preprocessing pipelines have been migrated. Collection storage and
raw-record parsing are converted and verified; `main.py` has not been created.

## Structure

```text
src/graph_ir/
    data/                # document/query/qrel structures and corpus loading
        preprocessing/   # source-specific document and query transformations
    indexing/            # postings and document/term mappings
    graphs/              # graph construction and weighting
    models/              # classical and graph-based retrieval
    spectral/            # clustering, contextual retrieval, query expansion
    embeddings/          # transformer encoding and embedding retrieval
    fusion/              # ranking combinations
    evaluation/          # standard and explicitly labeled legacy metrics
    experiments/         # experiment execution and result export
configs/                 # dataset, preprocessing, and experiment settings
collections/             # tracked CF inputs/raw records and small examples
notebooks/               # analysis and exploration workflows
tests/fixtures/          # small verification inputs and reference outputs
artifacts/               # generated files; ignored except the placeholder
to_merge/                # original local references; ignored by Git
main.py                  # future entry point
pyproject.toml           # package metadata and src-layout discovery
MERGE_PLAN.md            # inventory, migration scope, and acceptance gates
```

## Preprocessing and data

Each source project's preprocessing will be preserved as a named pipeline where
behavior differs. Document and query processing both belong in
`graph_ir.data.preprocessing`. Shared operations can be consolidated after their
outputs are verified. Corpus-format parsing and loading belong in `graph_ir.data`.

The `data` package contains Python code, not raw dataset files. Imported inputs
live in `collections/`; paths and formats are recorded in
[configs/collections.json](configs/collections.json).

Collections now use `corpus.jsonl`, `queries.jsonl`, and `qrels.tsv`, with explicit
IDs and direct paths in `configs/collections.json`. Cranfield has
explicit graded/binary relevance profiles; CF preserves its four assessor-specific
0..2 profiles alongside binary relevance. These prepare the data for nDCG. CF contains 1,239
documents and 100 queries; Cranfield contains 1,400 documents and 225 queries,
including two empty documents. NPL preserves its incomplete 1,430-document
snapshot and 93 queries without discarding missing-document judgments. The
`baeza` and `test` examples each have four documents and no supplied queries.

See [collections/README.md](collections/README.md) for record formats and
relevance score definitions. CF and Cranfield canonical records retain
raw text fields. Original source-project inputs remain under `to_merge/` for merge
reference; NPL/examples retain ordered tokens because raw prose is unavailable.
Migrated preprocessing writes generated outputs into `artifacts/`.

`artifacts/` is for reproducible outputs such as processed-corpus caches, indexes,
graphs, embeddings, rankings, and result spreadsheets. Original datasets and
irreplaceable historical results must have a separately documented storage home.

See [the merge plan](MERGE_PLAN.md) for the staged migration. The completed package
must support its workflows without imports or runtime data dependencies on
`to_merge/`.
