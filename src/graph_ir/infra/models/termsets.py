"""Query-dependent termset frequency matrices used by set-based models."""

from math import log2

import numpy as np

from graph_ir.data.preprocessing.collection import Collection


def calculate_tsf(collection: Collection, termsets: dict) -> np.ndarray:
    """Build a termset-by-document matrix using direct frequency lookups.

    termsets maps each nonempty termset to the IDs of documents containing all
    its terms, as produced by Apriori. Rows follow termset insertion order;
    columns follow collection.docs order. Frequencies use logarithmic weighting
    and rounding: round(1 + log2(minimum term frequency), 3).
    """
    matrix = np.zeros((len(termsets), collection.num_docs))

    for row, (termset, document_ids) in enumerate(termsets.items()):
        frequencies = [collection.inverted_index[term]["posting_list"] for term in termset]
        for document_id in document_ids:
            frequency = min(lookup[document_id] for lookup in frequencies)
            column = collection.doc_id_to_position[document_id]
            matrix[row, column] = round(1 + log2(frequency), 3)

    return matrix
