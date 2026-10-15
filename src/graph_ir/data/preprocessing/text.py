"""Small text transformations shared by documents and queries."""

from collections.abc import Iterable
import re
import warnings

import nltk
from nltk.tokenize import sent_tokenize

_TOKEN_PATTERN = re.compile(r"[^\W_]+")
_SENTENCE_END_PATTERN = re.compile(r"[.!?]")


def preprocess_text(text: str | list[str]) -> list[str]:
    """Lowercase text and split it into alphanumeric tokens.

    Punctuation and underscores separate tokens. Numbers, Unicode letters,
    accents, token order, and repeated terms are preserved. No stemming or
    lemmatization is applied. A token list is processed without modifying it.
    Stopword removal is a separate step -- see remove_stopwords.

    Args:
        text: Raw text or an existing token list.

    Returns:
        Processed tokens, or an empty list if no tokens remain.
    """
    if isinstance(text, list):
        text = " ".join(text)
    return _TOKEN_PATTERN.findall(text.lower())


def remove_stopwords(
    tokens: list[str], stop_words: Iterable[str], *, case_insensitive: bool = False,
) -> list[str]:
    """Filter tokens against a stopword set, independent of tokenization.

    Exact and case-sensitive by default: compose this after preprocess_text
    (which already lowercases) for case-consistent removal without needing
    this flag. Pass case_insensitive=True when tokens may not already be
    lowercase (for example, no preprocessor was applied) and stop_words is
    a lowercase set; only removal is case-insensitive -- surviving tokens
    keep their original case.

    Args:
        tokens: Tokens to filter, in order.
        stop_words: Tokens to exclude; pass a set for fast lookups.
        case_insensitive: Match tokens against stop_words ignoring case.

    Returns:
        The tokens not found in stop_words, in their original order.
    """
    if case_insensitive:
        stop_words = {word.lower() for word in stop_words}
        return [token for token in tokens if token.lower() not in stop_words]
    return [token for token in tokens if token not in stop_words]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt tokenizer.

    The 'punkt_tab' resource is downloaded on first use if missing, with a
    visible message -- never at import time, and never silently.

    Text with no sentence-ending punctuation (. ! ?) anywhere has no
    recoverable sentence boundaries -- it was likely stripped by an earlier
    preprocessing step -- and this function then just returns the whole
    text as one sentence. That degradation is otherwise silent, so it warns.

    The warning goes through the warnings module rather than print, so callers
    can escalate, silence or assert on it. Python's default "once per location"
    filter already keeps it from repeating for every document, without this
    module having to remember that it fired.

    Args:
        text: Raw text to split.

    Returns:
        Sentences in their original order.
    """
    if not _SENTENCE_END_PATTERN.search(text):
        warnings.warn(
            "text has no sentence-ending punctuation (. ! ?); "
            "it will be treated as a single sentence",
            stacklevel=2,
        )

    try:
        return sent_tokenize(text)
    except LookupError:
        print("nltk 'punkt_tab' resource not found; downloading it now...")
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(text)
