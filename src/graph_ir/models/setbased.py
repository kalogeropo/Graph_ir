from typing import Any
import numpy as np
from numpy import ndarray

from graph_ir.models.model import Model


class SetBasedModel(Model):
    """Set-based retrieval using IDF weighting on frequent termsets.

    No graph-based node weighting is applied. The model combines Apriori termset
    mining with cosine similarity ranking. Termsets are weighted by IDF alone,
    without graph properties.
    """

    def __init__(self, collection):
        super().__init__(collection)
        self._model = self.get_model()

    def get_model(self) -> str:
        return self.__class__.__name__

    def _model_func(self, freq_termsets: Any) -> ndarray:
        """Return uniform weights for termsets (no graph weighting).

        Pure set-based model uses IDF weighting only. This method returns ones
        so that the vectorizer applies IDF without additional graph-based scaling.

        Args:
            freq_termsets: Dictionary of frequent termsets from Apriori.

        Returns:
            Array of ones, one per termset.
        """
        return np.ones(len(freq_termsets))

    def _vectorizer(self, tsf_ij: ndarray, idf: ndarray, *args: Any) -> ndarray:
        """
        Applies IDF-based weighting to the termset frequency matrix.

        Args:
            tsf_ij (np.ndarray): Termset-document frequency matrix.
            idf (np.ndarray): Inverse document frequency vector.

        Returns:
            np.ndarray: Weighted document-term matrix.
        """
        return tsf_ij * idf.reshape(-1, 1)
