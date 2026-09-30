# Graph_ir complete repository merge plan

Status: scaffold aligned with this plan; migration stages remain proposed, based
on source inspection on 2026-09-28. No models or model preprocessing pipelines have
been executed or migrated; raw-record storage parsing has been converted and verified. Root `main.py` is not currently present; the first
usable workflow will introduce it.

Collection storage conversion completed: CF, Cranfield, NPL, `baeza`, and `test`
now have explicit-ID document/query JSONL and explicit relevance-profile TSV in tracked
`collections/`. Raw CF/Cranfield fields and assessments, legacy tokens/query text,
and source variants remain available in the original projects under `to_merge/`.
Migration scripts, manifests, duplicate archives, and generated verification
reports were removed after collection cleanup. Collection files were checked
against the raw inputs before cleanup; NPL remains incomplete and
Cranfield retains two empty documents. See [collection notes](collections/README.md),
[path configuration](configs/collections.json). Model reference runs and model
preprocessing migration remain pending.

Scope clarification: the final goal is one repository containing the functionality
of all four projects, including each project's preprocessing, parsers, indexing,
models, evaluation, experiments, exports, and analysis workflows. The model
inventory below is the starting point, not the full migration scope. Package
scaffolding now exists; implementation migration has not started.

This plan covers the merge only. Algorithm improvements, new retrieval methods,
parameter tuning, and performance optimization are outside its scope. Required
compatibility repairs and fixes that unblock migrated workflows are included,
but must be documented separately from behavior-preserving migration. Adding
standard metrics establishes a shared evaluation interface; it does not redefine
historical results or make improved effectiveness a merge acceptance condition.

`to_merge/` is ignored by Git as requested and remains a local source reference.
Migrated code must live in the tracked package and supporting directories. The
finished project must run without importing from or requiring `to_merge/`.

## 1. What is present

There are four source projects under `to_merge/`. Their similarly named models
are candidates for consolidation, not established equivalents. The shared model
files in `Gsb_model` and `GIRTE_Model` are not byte-identical.

### Gsb_model: classical retrieval, graph weighting, and fusion

Source directory: [models](to_merge/Gsb_model/models/).

| Implementation | Role | Proposed disposition |
| --- | --- | --- |
| `SetBased.py: SetBasedModel` | Apriori termsets with TF-IDF-style weighting | First retrieval model to migrate |
| `GSB.py: GSBModel` | Collection union graph, node weights, optional document k-core importance/pruning | Primary candidate for classical GSB; compare against other implementations |
| `WindowedGSB.py: WindowedGSBModel` | Fixed-size or proportional document windows, optional k-core weighting | Preserve window and tail-handling semantics explicitly |
| `GoW.py: Gow` | Graph-of-Words using `gowpy.TwidfVectorizer` | Separate retrieval model with optional dependency |
| `ΒΜ25.py: BM25Model` | Wrapper around `rank_bm25.BM25Okapi` | Baseline; migrate into ASCII-named `bm25.py` |
| `borda_count.py: BordaCount` | Combines two ranked lists | Move to ranking fusion rather than forcing termset-model inheritance |
| `DocGraph.py: DocGraph` | Intended document/word graph and document similarity graph | Prototype: `collect_info()` returns empty structures; `fit/evaluate` are stubs |
| `onlineGSB.py: onlineGSB` | Incremental collection/graph indexing experiment | Prototype, not a complete ranker; persistence is a stub |
| `Model.py: Model` | Shared termset representation and evaluation | Infrastructure, not a separate retrieval model |

The BM25 filename starts with Greek characters that resemble Latin `BM`.
The many `borda_*`, `maincore_*`, and window experiment scripts largely configure
these models; they should become experiment configurations after behavior is verified.

### GIRTE_Model: GSB fork plus transformer extensions

Source directory: [models](to_merge/GIRTE_Model/models/).

| Implementation | Role | Proposed disposition |
| --- | --- | --- |
| `GIRTE.py: GIRTEModel(tensors=False)` | GSB on BERT tokens using token-frequency adjacency | Expose as a distinct tokenized-GSB configuration |
| `GIRTE.py: GIRTEModel(tensors=True)` | Graph edges weighted by BERT token-embedding cosine similarity, with threshold `theta_val` | Preserve as GIRTE embedding-graph configuration |
| `Tensor.py: TensorModel` | Direct token-embedding retrieval: maximum document-token similarity per query token, then mean | Separate experimental embedding baseline; normalize its result API |
| `GIRTE_old.py: GIRTEModel` | Older alternative implementation | Retain as historical reference until differences are documented |
| `GSB`, `WindowedGSB`, `SetBased`, `GoW`, `BM25`, `BordaCount`, `Model` files | Forked classical implementations | Compare with `Gsb_model`; do not copy a second public classical API |

`Preprocess/Tok_Document.py` and `Tok_Collection.py` are essential supporting
components. BERT base/large, stopword settings, and frequency/embedding adjacency
are configurations rather than independent codebases.

### Spectral-Clustering-and-Query-Expansion-using-Embeddings-on-the-GSB

Source directory: [infre/models](to_merge/Spectral-Clustering-and-Query-Expansion-using-Embeddings-on-the-GSB/infre/models/).

| Implementation | Role | Proposed disposition |
| --- | --- | --- |
| `vs.py: VectorSpace` | TF-IDF vector-space baseline | Retain as a separate baseline |
| `sb.py: SetBased` | Set-based retrieval | Compare with classical SB before consolidation |
| `gsb.py: GSB` | Graphical set-based retrieval | Compare graph weights and scores with classical GSB |
| `gsbw.py: GSBWindow` | Fixed/proportional window GSB | Compare segmentation and weighting with `WindowedGSBModel` |
| `pgsb.py: PGSB` | Spectral clustering followed by graph pruning and node reweighting | Retain research variant |
| `pgsbw.py: PGSBW` | Windowed clustered/pruned GSB | Retain research variant |
| `cgsb.py: ConGSB` | Cluster-derived contextual weights and query-expansion methods | Retain, with expansion configured separately |
| `cgsbw.py: ConGSBWindow` | Windowed contextual GSB | Retain research variant |
| `bmodel.py: BaseIRModel` | Termset retrieval and evaluation infrastructure | Compare and consolidate common behavior |

The embeddings here come from spectral graph embedding, unlike GIRTE's BERT
embeddings. Do not treat these as interchangeable embedding implementations.
`infre/tools/sc.py` wraps spectral embedding and clustering and returns both labels
and embeddings. Query expansion includes nearest-neighbor and cluster-centroid
paths. Constructor choice alone does not establish which expansion ran.

### Graphical-Set-based-model: procedural predecessor

Entry point: [main_v3.py](to_merge/Graphical-Set-based-model/main_v3.py).
Support: `k_core_modules.py` and `graph_creation_scripts.py`.

This project contains procedural SB/GSB calculations and indexing branches for
maincore, whole-document, percentage, dot/sentence, constant-window, and
sentence/paragraph variants. It also includes a Vazirgiannis-style adjacency
helper. Labels in old comments and result variables are inconsistent with some
active branches, so names alone are insufficient to identify algorithms.

Keep it as a reference during the main merge. Later, trace each branch and port
only behavior not already represented, including any distinct penalties or
sentence/paragraph segmentation. Do not assume all its algorithms are redundant.

## 2. Decisions that must precede consolidation

1. **Document identity and preprocessing.** Both model bases index matrices using
   `doc_id - 1`. `Gsb_model` can use the maximum document ID as `num_docs`, while
   `infre` starts from the number of directory entries. Missing IDs therefore
   affect dimensions and potentially IDF. Introduce explicit ID-to-row mappings,
   preserve original IDs, and record tokenization, casing, stopwords, and corpus
   selection for every comparison. Keep source-specific preprocessing profiles
   until their differences are measured.
2. **Metrics.** `utilities/document_utls.py: calc_precision_recall` and
   `infre/metrics.py: precision_recall` average precision/recall at relevant hits;
   these are not ordinary precision@k and recall@k. The former also increments
   its rank counter before testing the cutoff, producing an off-by-one cutoff
   for typical k values. The latter can return NaN when no relevant item is hit.
   Preserve historical metrics under explicit legacy names; add separately
   validated P@k, R@k, AP/MAP, MRR, and nDCG@k. Do not relabel old results as new metrics.
   CF's four-digit assessment codes are four separate 0..2 ratings. Preserve
   assessor-specific profiles; any aggregate requires an explicit policy.
   Cranfield's graded profile preserves 1..4 and maps -1 to zero; its corrected
   binary profile excludes those 225 nonrelevant pairs. Keep the historical
   all-listed-pairs profile for reproduction. nDCG must record the grade profile,
   cutoff, gain function, unjudged policy, ties, and zero-IDCG behavior.
3. **Model state.** GSB implementations write `nwk` into the collection's inverted
   index. Other model variants can overwrite those weights. Use isolated legacy
   collections for reference runs and model-owned weights in the merged code.
   Repeated fit/evaluate calls must not accumulate stale outputs.
4. **APIs.** Classical models use `fit(queries, min_freq)` and `evaluate(k)`;
   `infre` uses `fit(queries, mf)` and `evaluate(relevant)`, with additional
   `fit_evaluate` paths. `TensorModel.fit()` prints metrics and returns `None`.
   Separate retrieval results from evaluation before building a unified runner.
5. **Imports and dependencies.** Generic absolute imports such as `models` and
   `Preprocess` collide across projects. Requirements disagree on NumPy,
   NetworkX, pandas, and scikit-learn versions. The spectral collection imports
   `gensim`, which its requirements omit; the GSB BM25 module attempts to install
   its missing dependency during import. Build and verify a new dependency set;
   do not concatenate the old requirements files or import all projects together.
6. **Paths and caches.** GIRTE writes to `C:/picklejar`; its matrix cache name
   includes BERT/stopword/method choices but not collection identity. Move writes
   into explicit artifact directories and key caches by corpus, preprocessing,
   model revision, and relevant configuration. Constructors should not silently
   download models or write artifacts.
7. **Spectral reproducibility.** Helpers randomly connect disconnected components;
   spectral embedding and KMeans have no exposed seed. Node order must be mapped
   explicitly to embedding rows. Audit pruning before migration: it removes
   edges while iterating and can reference an unset `cond` for an empty condition.
8. **Parameter semantics.** Existing GSB interprets integer and float `h_val` and
   `p_val` differently. Window tails and proportional-window rounding also matter.
   Record legacy behavior, then expose explicit parameters with compatibility
   conversions rather than silently changing experiment meaning.

These findings are from reading source, not verified runtime failures.

## 3. Repository structure

Use a small `graph_ir` package, borrowing working behavior from both classical
implementations and the modular organization of `infre`. No source project is
assumed correct in its entirety. The package directories below now exist as
scaffolding. Introduce implementation modules as their migration stage needs them;
do not build a general plugin framework first.

```text
main.py                  # thin entry point once the first model is usable
pyproject.toml
src/graph_ir/
    data/                # document/query/qrel structures and corpus parsers/loaders
        preprocessing/   # each source pipeline and shared text operations
    indexing/            # postings and explicit document/term mappings
    models/              # SB, BM25, VS, GSB, windowed GSB, GoW
    graphs/              # shared graph construction/weighting as overlap emerges
    spectral/            # PGSB, PGSBW, contextual variants, expansion
    embeddings/          # reusable BERT encoder, tokenized GSB, GIRTE, Tensor
    fusion/              # Borda over ranked results
    evaluation/          # standard metrics and labeled legacy metrics
    experiments/         # configuration-driven runner and model registry
tests/fixtures/          # tiny corpus, queries, qrels, reference outputs
configs/                 # reproducible experiment settings
collections/             # selected input snapshots and original CF records
notebooks/               # destination for migrated analysis workflows
artifacts/               # generated results/caches, excluded from source control
to_merge/                # local migration references, excluded from source control
```

### Why these directories exist

All package directories belong to one `graph_ir` library in one repository.
They separate responsibilities, not source projects or installations.

- `data/` contains Python code for representing and reading documents, queries,
  and relevance judgments. It is not a directory for storing raw datasets.
- `data/preprocessing/` will hold the preprocessing from every source project:
  text cleaning, case handling, stopword policies, tokenization, and segmentation.
  Keep named source-compatible pipelines where behavior differs; share individual
  operations only after equivalence is established. Document and query processing
  must both be covered, including intentional differences between them.
- `indexing/` turns processed documents into postings and ID mappings; `graphs/`
  constructs and weights graphs from those representations.
- `models/`, `spectral/`, `embeddings/`, and `fusion/` hold retrieval algorithms,
  their specialized support, and ranking combinations. BERT tokenization must be
  coordinated with the preprocessing pipeline and encoder configuration.
- `evaluation/` measures results; `experiments/` runs configured workflows;
  `configs/` records their settings; `notebooks/` preserves analysis workflows.
- `artifacts/` stores generated outputs: processed-corpus caches, indexes, saved
  graphs, embeddings, rankings, and result spreadsheets. These can be large and
  regenerated, so they are ignored by Git. It must not be the only home of source
  code, original datasets, or irreplaceable historical results.
- `tests/fixtures/` holds small tracked inputs and expected outputs that verify
  preprocessing and retrieval behavior.

Imported inputs are tracked under `collections/`, with paths/formats in
`configs/collections.json` and provenance/checksums beside the data. The Python
`data/` package contains the loading and processing code. Inventory additional
datasets and historical results before choosing their tracked or external storage.
Preserve reproducible access to them as part of the repository workflow.

### Preprocessing and supporting-code coverage

Before migrating each model family, inventory and migrate its complete input path:

| Source | Initial inspection targets |
| --- | --- |
| `Gsb_model` | `Preprocess/Document.py`, `Preprocess/Collection.py`, `parse.py`, `utilities/parsers.py`, and preprocessing helpers in `utilities/document_utls.py` |
| `GIRTE_Model` | Classical preprocessing plus `Preprocess/Tok_Document.py`, `Preprocess/Tok_Collection.py`, token/embedding alignment, stopwords, and truncation behavior |
| Spectral project | `infre/preprocess/`, dataset parsing notebooks, and document/query loading and normalization |
| Procedural predecessor | `preprocces.py`, preprocessing in `k_core_modules.py`, segmentation in `graph_creation_scripts.py`, and input handling in `main_v3.py` |

Extend this list by tracing callers and scripts; it is not yet a complete function
inventory. Maintain a source-to-destination migration checklist covering code,
configurations, notebooks, datasets, and result provenance. Every item must have
a destination, a verified equivalent, or an explicitly agreed historical status.
An unfinished prototype remains visible as unfinished; it is not silently dropped.

For each pipeline, compare original and migrated outputs on fixed inputs: document
and query IDs, ordered tokens, frequencies, sentence/window boundaries, and BERT
token alignment where applicable. Include punctuation, casing, empty text,
stopwords, repeated words, and long documents. Model ranking parity alone is not
enough to demonstrate preprocessing parity.

The first proposed compatibility profile uses the spectral CF legacy token
snapshot in the spectral source project's `collections/CF/docs/` unchanged.
Capture required reference fixtures in tracked tests during model migration.
Preserve the spectral query loader's two paths: its default Gensim cleaning plus
NLTK English stopword removal, and whitespace splitting with `prep=False`. Do not
silently remove document stopwords or assume document/query processing is identical.
Migrated raw-input profiles should parse `TI`, `AB`, and `EX` fields from
the raw fields in `collections/cf/corpus.jsonl` before transforming text. Preserve each source workflow's
field selection, punctuation/boundaries for windowed models, and original text
for transformer encoding. Do not introduce new preprocessing experiments as
part of the merge. The old parser notebook is not a
validated preprocessing implementation; its `('AB' or 'EX')` test misses `EX`.

Proposed public flow: `model.index(corpus)` then `model.search(queries, top_k)`
returning query IDs, document IDs, scores, and ranks. Evaluation consumes this
result plus qrels separately. Termset models still perform query-dependent
Apriori/vector construction during search; they need not fit a query-independent
document matrix. Fusion consumes rankings rather than requiring fake vectorizers.

Keep transformer and GoW dependencies optional. Establish supported Python and
package versions through environment checks and smoke tests during implementation.

## 4. Migration stages and completion gates

Default order: reproducible classical models first, spectral variants next,
transformer extensions afterward. A different research priority can reorder the
last stages without skipping the shared data/evaluation foundation.

| Stage | Work | Gate before proceeding |
| --- | --- | --- |
| 0. Preserve and inventory sources | Preserve recoverable source snapshots plus hashes, dependency environments, dataset provenance, parameters, and entry points. Create a source-to-destination checklist for all four projects. Capture isolated reference outputs per family before its migration; record blockers and any minimal reference repairs. | All sources are preserved and inventoried; the first family has reference outputs or documented blockers. Later families do not block the first migration; no reference is invented |
| 1. Parsing, preprocessing, indexing, and evaluation | Implement document/query/qrel structures, explicit ID mappings, loaders and required raw parsers, named document/query preprocessing profiles, postings, a tiny fixture, migrated legacy metrics, and independently validated standard metrics. Extend these components with each family's source-specific requirements. | Parser fields and IDs, ordered tokens, frequencies, and boundaries match references or documented repairs; hand-calculated metrics pass; ranking and metric edge cases have explicit conventions |
| 2. First usable model | Port SB with Apriori; add the thin CLI in root `main.py`. | A tiny corpus runs end to end; termsets, frequencies, scores, and rankings agree with the chosen reference or documented fixes |
| 3. Baselines | Add BM25 and VectorSpace with explicit document order and standard result objects. | Fixed-input rankings are reproducible; scores/IDs and standard metrics are validated |
| 4. Graph models | Add GSB, fixed/proportional window GSB, then k-core/pruning settings. Compare GSB and infre implementations. | Compare adjacency, union graph, Win/Wout, node weights, and ranking; running models in either order gives the same results |
| 5. GoW and fusion | Add the GoW adapter and Borda fusion; translate representative ensemble scripts to configs. | Validate different ranking lengths, ties, missing documents, and repeated calls; compare representative legacy ensembles |
| 6. Spectral family | Add seeded embedding/clustering, PGSB/PGSBW, then ConGSB/ConGSBWindow and expansion strategies. | Preserve term-to-embedding alignment; validate disconnected graphs and pruning; verify migrated expansion outputs against references and seeded reproducibility |
| 7. Transformer family | Reuse one configured encoder; add tokenized GSB, embedding GIRTE, then TensorModel. Replace hard-coded caches. | Token-only behavior is validated first; caches cannot cross corpora; truncation/chunking, token aggregation, thresholding, and tensor order are tested |
| 8. Complete repository coverage | Trace procedural branches, preserve unique preprocessing and algorithms, assess DocGraph/onlineGSB, and migrate remaining parsers, exports, experiment scripts, and analysis workflows. Consolidate duplicates only after parity evidence. | The source-to-destination checklist is accounted for; supported workflows run from raw inputs through preprocessing to results without `to_merge/`; any remaining deferral is explicit and means the full merge is still incomplete |

For each stage, use one small reviewable change or commit per model/behavior.
Record bug fixes separately from mechanical migration, especially changes that
alter historical rankings or metrics. Retain the reference implementation until
the replacement passes its gate. Each model stage includes its required original
preprocessing and supporting workflow, not just the model class. No deletion of
legacy trees is part of this plan.

## 5. Merge verification strategy

- Start with a tiny hand-inspectable corpus, then a pinned CF subset, then full CF.
  Use the imported spectral CF snapshot. The GSB/GIRTE copies lack 30 documents;
  29 of those IDs are referenced by 166 judgments. Rankings across these snapshots
  are not directly comparable. Cranfield has two empty documents and NPL is
  incomplete in the inspected copies. Both are now stored with these issues
  recorded; NPL is not a complete evaluation benchmark.
- Record document/query/qrel hashes, preprocessing profile, model configuration,
  dependency versions, seed, and source version with each result.
- Compare exact IDs and rankings, with declared tolerances for floating-point
  scores; define deterministic ties. Inspect intermediate graph weights when
  rankings differ. Spectral embeddings can differ by sign/rotation, so compare
  relevant similarities and behavior rather than raw coordinates alone.
- Keep legacy reproduction and standardized evaluation as separate reports.
  Report migration parity and documented differences; improved retrieval quality
  is not a merge gate.
- Run representative source workflows through parsing/loading, preprocessing,
  indexing, retrieval, evaluation, export, and notebook analysis. Verify output
  schemas and provenance as well as rankings. Full transformer verification
  follows small smoke runs; performance optimization belongs to later work.

## 6. Recommended first implementation increment

Preserve the source snapshots and create the migration checklist first. Establish
an isolated SetBased reference: the current GSB `SetBasedModel._model_func()`
raises `NotImplementedError` when inherited `fit()` calls it, so document its
blocker and any minimal reference repair rather than claiming an untouched run.

Use the existing package scaffold, add the fixture, migrate the required document
and query loading/preprocessing, explicit-ID indexing and Apriori behavior, and
legacy metrics. Verify standard metrics independently, port SetBased, and create
`main.py` to run that complete workflow.
Keep its acceptance scope small: explicit document IDs, reproducible scores and
rankings, source-parity evidence, and no unrelated model migration.

The repository scaffold and selected collection snapshots are in place. Original
files under `to_merge/` remain migration references. Root `main.py` will be
created during implementation. Parsers, preprocessing, indexing, metrics, models,
exports, experiments, and analysis migration remain implementation work.

## 7. Definition of a completed merge

The merge is complete only when:

- All four projects' code and workflows are accounted for in the migration
  checklist, including preprocessing, parsers, indexing, graph helpers, models,
  metrics, experiment settings, exports, notebooks, datasets, and historical results.
- Every implemented source behavior has a tracked destination or a verified
  equivalent. Distinct variants remain available through explicit configurations;
  unfinished prototypes retain an explicit historical status without being
  presented as completed models.
- Document and query preprocessing preserve source-specific semantics, including
  segmentation and transformer token alignment. Required raw-input and existing
  tokenized-input workflows both run from documented inputs.
- Legacy metrics retain explicit names and provenance. Standard metrics have
  validated definitions covering cutoffs, ties, duplicates, short result lists,
  no-hit queries, queries without relevant documents, and AP denominators.
- Representative workflows run through retrieval, evaluation, exports, and
  analysis using only tracked package code and documented data/dependencies,
  without runtime access to `to_merge/`.
- Reference comparisons and all required compatibility repairs are documented.
  Any deferred implemented source functionality means the full merge remains
  incomplete. New algorithms, tuning, and optimization are separate future work.
