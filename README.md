# Graph_ir

One Python package for the complete retrieval workflows from the four projects
under `to_merge/`: parsing, preprocessing, indexing, models, evaluation,
experiments, and analysis.

The repository contains package scaffolding and validated collection snapshots.
No models or preprocessing pipelines have been migrated, and `main.py` is still
the starter script.

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

The selected spectral CF snapshot has 1,239 documents, 100 queries, and 4,819
relevance judgments. Original CF records are preserved in `collections/CF/raw/`.
The `baeza` and `test` examples each have four documents and no supplied queries
or judgments. Cranfield and NPL imports are pending data-quality issues.

See [collections/README.md](collections/README.md) for validation evidence, source
provenance, checksums, and the preprocessing proposal. Existing CF documents are
already tokenized: the first compatibility pipeline should load them unchanged.
New text-processing pipelines should start from the preserved raw records and
write derived outputs into `artifacts/`.

`artifacts/` is for reproducible outputs such as processed-corpus caches, indexes,
graphs, embeddings, rankings, and result spreadsheets. Original datasets and
irreplaceable historical results must have a separately documented storage home.

See [the merge plan](MERGE_PLAN.md) for the staged migration. The completed package
must support its workflows without imports or runtime data dependencies on
`to_merge/`.
