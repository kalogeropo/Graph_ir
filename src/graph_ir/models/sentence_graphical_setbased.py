from graph_ir.data.preprocessing.text import split_sentences
from graph_ir.models.graphical_setbased import GSBModel


def sentence_windows(document, preprocessor):
    """Split a document into per-sentence token windows.

    Sentence boundaries come from NLTK's sentence tokenizer applied to the
    document's original text. A document with no recoverable raw text
    (pre-split tokens only) falls back to one window spanning the whole
    document.

    Args:
        document: The document to split.
        preprocessor: Tokenizer applied to each sentence (typically the
            collection's preprocessor, for vocabulary-consistent tokens).
            Falls back to str.split when None.

    Returns:
        One token list per sentence, or a single list of document.terms
        when there is no raw text to split.
    """
    if isinstance(document.text, str) and document.text:
        tokenize = preprocessor or str.split
        return [tokenize(sentence) for sentence in split_sentences(document.text)]
    return [document.terms]


class SentenceGSBModel(GSBModel):
    """GSB variant whose co-occurrence windows are sentences.

    See sentence_windows for how sentence boundaries are determined.
    """

    def __init__(self, collection, h_val: int = 1, k_core_bool: bool = False):
        super().__init__(collection, k_core_bool=k_core_bool, h_val=h_val)

    def _window_sets(self, document):
        """One window per sentence; see sentence_windows for the boundaries."""
        return [(1, sentence_windows(document, self.collection.preprocessor))]
