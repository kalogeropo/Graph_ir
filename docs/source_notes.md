# Source Project Findings

These notes are from reading the four source projects under `to_merge/` to understand their structure, algorithms, and quirks before migration.

## Model families

### SetBased (SB): the foundation
Retrieval runs per query. `Model.fit` runs Apriori to find frequent termsets. Each termset has an IDF and per-document frequency. Documents are ranked by cosine similarity against the query vector. Rest of the models only change weights and vectorization.

### GSB (graph set-based)
Each document becomes a term co-occurrence matrix. These are summed into one union graph. Each term gets a weight `nwk` from graph properties. A termset's weight is the product of its terms' `nwk`. Warning: `nwk` is written into the shared inverted index, so models can overwrite each other's state.

### Windowed GSB
Same as GSB but the matrix is built per window of the document instead of per whole document.

### Spectral family (`infre`)
- **PGSB:** clusters the union graph, then prunes edges and recomputes `nwk`.
- **ConGSB:** uses embeddings for contextual weights and query expansion.
- **W variants:** windowed versions.

### GIRTE (BERT)
- **`tensors=False`:** same GSB on BERT word-piece tokens.
- **`tensors=True`:** edge weights are BERT token embedding cosine similarity with threshold.
- **`TensorModel`:** ignores the graph. Slow: one cosine call per query-token / document-token pair.

### Fusion and other pieces
- **`BordaCount`:** merges two rankings.
- **BM25, VectorSpace, GoW:** baselines.
- **`DocGraph`, `onlineGSB`:** unfinished stubs.

## Decisions that must precede consolidation

1. **Document identity and preprocessing.** Both model bases index matrices using `doc_id - 1`. `Gsb_model` can use the maximum document ID as `num_docs`, while `infre` starts from the number of directory entries. Missing IDs affect dimensions. Introduce explicit ID-to-row mappings, preserve original IDs, and record tokenization, casing, stopwords, and corpus selection for every comparison.

2. **Metrics.** `utilities/document_utls.py: calc_precision_recall` and `infre/metrics.py: precision_recall` average precision/recall at relevant hits; these are not ordinary precision@k and recall@k. The former increments its rank counter before testing the cutoff, producing an off-by-one for typical k values. The latter can return NaN when no relevant item is hit. Preserve historical metrics under legacy names; add validated P@k, R@k, AP, MRR, nDCG.

3. **Model state.** GSB implementations write `nwk` into the collection's inverted index. Other variants can overwrite those weights. Use isolated legacy collections for reference runs and model-owned weights in merged code.

4. **APIs.** Classical models use `fit(queries, min_freq)` and `evaluate(k)`. `infre` uses `fit(queries, mf)` and `evaluate(relevant)`, with `fit_evaluate` paths. `TensorModel.fit()` prints metrics and returns `None`. Separate retrieval results from evaluation before building a unified runner.

5. **Imports and dependencies.** Generic absolute imports like `models` and `Preprocess` collide. Requirements disagree on NumPy, NetworkX, pandas, scikit-learn. Spectral imports `gensim` but omits it from requirements. GSB BM25 tries to install its missing dependency during import. Build and verify a new dependency set.

6. **Paths and caches.** GIRTE writes to `C:/picklejar`. Matrix cache names include BERT/stopword/method but not collection identity. Constructors should not silently download or write artifacts.

7. **Spectral reproducibility.** Helpers randomly connect disconnected components. Spectral embedding and KMeans have no exposed seed. Node order must be mapped to embedding rows. Pruning removes edges while iterating and can reference unset `cond`.

8. **Parameter semantics.** Integer vs. float `h_val` and `p_val` mean different things. Window tail handling and rounding also matter. Record legacy behavior and expose explicit parameters.

## Known optimization targets (profile before touching)
- `calculate_tsf`: posting list scans per term per termset
- `WindowedGSB.doc_to_matrix`: O(windows × V²) term pair loops
- `BordaCount`: pandas lookups are quadratic
- `TensorModel`: per-pair cosine calls (use per-document matmul)
- Dense per-document matrices in `union_graph` construction
- Per-query stored matrices in Model (store results instead)

## Known implementation bugs and quirks

| Source | Issue |
|---|---|
| `SetBasedModel` | `_model_func` not implemented; raises `NotImplementedError` |
| `Collection.Collection` | ID fallback to 696969 if filename has no digits |
| `Collection.create_collection` | Uses `max(doc_ids)` as `num_docs`, not actual count |
| `Document.split_document` | Hard-coded minimum window of 7; drops remainder silently |
| `GSBModel.kcore_nodes` | Empty graph can crash with ValueError |
| `GIRTE.union_graph` | Caches to hard-coded `C:/picklejar` |
| `TensorModel.fit` | Prints progress inside fit; returns None instead of self |
| `infre.prune_graph` | Can reference unset `cond` variable |
| All models | Queries are printed during fit |
| Query/document handling | No case normalization; uppercase docs vs. mixed-case queries cause silent score-zero |

## Pre-migration verification needed
- Whether queries and documents in CF match in case
- Whether stopword sets are consistent across projects
- Whether BERT truncation and token alignment are documented
- Whether the old SetBased can run on baeza and test (minimal fix for `_model_func`)
