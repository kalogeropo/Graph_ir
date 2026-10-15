class Document:
    """Represent a document as an ordered sequence of terms.

    Args:
        text: Raw text to split on whitespace, or an existing list of tokens.
            Token case, punctuation, order, and repetition are preserved.
        id: Document identifier. Defaults to -1 when unspecified.

    Attributes:
        text: Original input text or token list.
        id: Document identifier.
        terms: Tokens used for indexing. Initially copied or split from text;
            a collection may replace them with preprocessed tokens.
        num_of_words: Number of tokens in terms.
    """

    def __init__(self, text: str | list[str] = "", id: int | str = -1):
        self.text = text
        self.id = id
        self.terms = self.tokenize()
        self.num_of_words = len(self.terms)

    def tokenize(self) -> list[str]:
        """Return a copy of the input tokens, or split raw text on whitespace."""
        if isinstance(self.text, list):
            return self.text.copy()
        return self.text.split()
