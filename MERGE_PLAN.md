# Graph_ir: merge plan

## Goal
One `graph_ir` package containing the workflows of the four projects in `to_merge/` (`Gsb_model`, `GIRTE_Model`, Spectral `infre`, `Graphical-Set-based-model`): preprocessing, indexing, models, evaluation, experiments. It must run without `to_merge/`.

## Rules
1. **Parity first, then optimize.** Port behavior, verify against saved reference output, commit. Only then optimize, one change per commit, re-verified. Record before/after timings in the commit message.
2. **Reference outputs are fixtures.** Run the old code in a throwaway venv (old library versions), on `baeza`, `test`, then CF. Save rankings and scores under `tests/fixtures/` with the library versions recorded.
3. **Bug fixes and optimizations are separate commits** from migration, with a note when rankings change.
4. **No prints, file writes or downloads in constructors.** Model state lives on the model, not in the collection's index. Seed anything random.
5. **Uniform API:** `model.index(collection)`, `model.search(queries, top_k)` returning IDs, scores, ranks. Evaluation is separate.

## Layout
```
src/graph_ir/
  data/               corpus/query/qrel loading, Document, Collection
    preprocessing/
  indexing/           inverted index, ID-to-row mapping
  models/             SB, BM25, VS, GSB, windowed GSB, GoW
  graphs/  spectral/  embeddings/  fusion/  evaluation/  experiments/
tests/  tests/fixtures/  configs/  collections/  notebooks/  artifacts/
```
`to_merge/` stays local and git-ignored.

## Environment
`uv`, Python >= 3.13 (matches lockfile), `uv.lock` committed. Core: numpy, networkx, scipy. Optional groups: `dev` (pytest, ruff), `spectral` (gensim, scikit-learn), `embeddings` (torch, transformers), `gow`. Reference venv for old code is separate and untracked.

## Stages

| # | Work | Done when |
|---|---|---|
| 0 | Commit current work; remove debug prints; add `dev` group; run tests. Reference-run old SetBased (minimal fix for missing `_model_func`) on baeza/test/CF; save fixtures. Back up raw CF/Cranfield. | Tests pass; fixtures saved; raw data backed up |
| 1 | Flatten `infra/`; move index to `indexing/`; drop duplicate `calculate_tsf`; make stopwords lazy/explicit; fix query-ID fallback. | Tests unchanged and green |
| 2 | SetBased model + `main.py`. Compare with fixtures; verify Apriori/tsf against them. | Rankings and scores match reference |
| 3 | Profile CF run; optimize top hotspots one at a time. | Same rankings, timings recorded |
| 4 | BM25, VectorSpace. | Fixed-input rankings reproducible |
| 5 | GSB, windowed GSB, k-core/pruning; compare with `infre` versions. | Adjacency, node weights, rankings match; run order irrelevant |
| 6 | GoW, Borda fusion. | Ties, unequal lengths, repeated calls tested |
| 7 | Spectral family (PGSB, ConGSB, expansion), seeded. | Term-to-embedding alignment kept; results reproducible |
| 8 | GIRTE / Tensor (BERT), keyed caches, no `C:/picklejar`. | Token-only path verified first; caches cannot cross corpora |
| 9 | Procedural predecessor branches, DocGraph/onlineGSB status, remaining scripts/notebooks. | Every checklist item has a destination or agreed status |

## Known risks to verify
Query/doc case mismatch; legacy metric cutoff off-by-one (kept, labeled legacy); `h_val`/`p_val` int-vs-float meaning; shared `nwk` state; window tail handling; NaN in `infre` metrics; unseeded spectral clustering.

## Metrics
Keep legacy metrics under legacy names; add validated P@k, R@k, AP, MRR, nDCG with hand-checked tests.

## Definition of done
All four projects in a source-to-destination checklist (`docs/checklist.md`), each item migrated, verified, or explicitly marked historical. Representative workflows run from raw inputs to results with tracked code only.

See `docs/source_notes.md` for detailed findings from the original code.
