"""Document and query preprocessing pipelines for all source projects.

Preserve the distinct GSB, GIRTE, infre, and procedural pipelines during migration.
Share cleaning, tokenization, stopword, and segmentation operations only after
their behavior has been compared. The optional text preprocessing function is a
new experiment, rather than a reproduction of any original pipeline.
"""
