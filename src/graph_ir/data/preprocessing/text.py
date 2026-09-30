"""Small text transformations shared by documents and queries."""

from collections.abc import Container
import re


_TOKEN_PATTERN = re.compile(r"[^\W_]+")


def preprocess_text(
    text: str | list[str], *, stop_words: Container[str] = (),
) -> list[str]:
    """Lowercase text and split it into alphanumeric tokens.

    Punctuation and underscores separate tokens. Numbers, Unicode letters,
    accents, token order, and repeated terms are preserved. No stemming or
    lemmatization is applied. A token list is processed without modifying it.

    Args:
        text: Raw text or an existing token list.
        stop_words: Lowercase tokens to exclude; pass a set for fast lookups.
            The default keeps all tokens.

    Returns:
        Processed tokens, or an empty list if no tokens remain.
    """
    if isinstance(text, list):
        text = " ".join(text)
    return [
        token for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in stop_words
    ]
