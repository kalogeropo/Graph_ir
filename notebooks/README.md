# Analysis notebooks

Destination for analysis and dataset-exploration notebooks migrated from all four
source projects. No notebooks have been migrated yet.

Reusable parsing, preprocessing, retrieval, and evaluation code belongs in
`src/graph_ir/`; notebooks should call that package. Record input provenance and
experiment settings, and write generated outputs to `artifacts/`.

The selected CF collection and its original records are now available under
`collections/CF/`. See [the collection notes](../collections/README.md) and
[collection configuration](../configs/collections.json). The legacy parsing
notebook has not been migrated; its parsing behavior needs correction and
verification before reuse.
