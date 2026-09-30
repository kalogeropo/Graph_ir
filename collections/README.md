# Collections

Each collection is stored in its own folder:

- `corpus.jsonl`: one document per line, with an explicit `_id`.
- `queries.jsonl`: one query per line, with `_id` and `text`.
- `qrels.tsv`: relevance judgments with columns `query-id`, `corpus-id`, and `score`.
- `assessments.jsonl`: original relevance codes, available for CF and Cranfield.

Paths and relevance profiles are listed in
[configs/collections.json](../configs/collections.json).

| Collection | Documents | Queries | Judgment pairs | Notes |
| --- | ---: | ---: | ---: | --- |
| cf | 1,239 | 100 | 4,819 | Raw title, abstract, and extract fields |
| cranfield | 1,400 | 225 | 1,837 | Two empty documents: 471 and 995 |
| npl | 1,430 | 93 | 2,083 | Incomplete: 1,839 judgments reference missing documents |
| baeza | 4 | 4 | 16 | Synthetic test judgments based on token a |
| test | 4 | 0 | 0 | Example with ordered tokens |

CF and Cranfield preserve raw text fields. NPL and the examples preserve existing
ordered tokens. The test collection has an empty query file and header-only qrels.
Baeza has four identical example queries. Its synthetic binary judgments score
documents 1 and 2 as relevant because they contain token `a`; documents 3 and 4
score zero. These judgments are for testing, not an externally assessed benchmark.

## Relevance scores

**CF:** each original code contains four assessor ratings. For example, `1222`
means `[1, 2, 2, 2]`. Each rating is 0=not relevant, 1=marginally relevant,
or 2=highly relevant. `qrels.tsv` is binary: relevant if any assessor gives a
positive rating. `qrels-assessor-1.tsv` through `qrels-assessor-4.tsv` preserve
the individual grades for nDCG. No combined grade is assumed.
[Source description](https://people.ischool.berkeley.edu/~hearst/irbook/cfc.html)

**Cranfield:** `qrels.tsv` preserves grades 1–4, with 4 strongest, and maps raw
-1 to 0. `qrels-binary.tsv` contains 1 for positive grades and 0 otherwise.
There are 1,612 relevant and 225 nonrelevant pairs. Original -1 values remain
in `assessments.jsonl`. Query IDs follow query order to match the judgments;
`source_id` retains the original raw query ID.
[Grade definitions](https://ir-datasets.com/cranfield.html)

NPL has binary judgments and remains an incomplete benchmark.
Generated preprocessing outputs belong in `artifacts/preprocessing/`.
