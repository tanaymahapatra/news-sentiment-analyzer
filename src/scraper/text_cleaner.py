"""Light normalization of scraped article text before DistilBERT inference.

DistilBERT was pretrained on natural language, so this module deliberately
does NOT do classical-ML preprocessing: no lowercasing, no stopword removal,
no stemming, no lemmatization. It only removes artifacts introduced by HTML
scraping, and leaves the actual language untouched.

Public API:
    clean_text(text) -> str
"""

from __future__ import annotations

import re
import unicodedata

# Invisible characters that survive HTML extraction.
ZERO_WIDTH = "\u200b\u200c\u200d\ufeff\u00ad"

# Typographic characters mapped to their plain ASCII equivalents, because the
# DistilBERT WordPiece vocabulary handles ASCII punctuation more reliably.
CHAR_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ",
}

_TRANSLATION_TABLE = str.maketrans(CHAR_REPLACEMENTS)


def _strip_control_chars(text: str) -> str:
    """Drop control/format characters but keep newlines and tabs."""
    for char in ZERO_WIDTH:
        text = text.replace(char, "")
    return "".join(
        ch for ch in text
        if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, trim lines, cap blank lines at one."""
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_punctuation_spacing(text: str) -> str:
    """Remove the stray space before punctuation that HTML extraction adds.

    Example: "said Prinstein , who is" -> "said Prinstein, who is"
    """
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def clean_text(text: str) -> str:
    """Normalize scraped article text for DistilBERT.

    Preserves casing, punctuation, negations, numbers and sentence structure.

    Args:
        text: Raw article text from the scraper.

    Returns:
        Cleaned text. Returns an empty string if the input is blank.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text() expects a string, got {type(text).__name__}.")

    if not text.strip():
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_TRANSLATION_TABLE)
    text = _strip_control_chars(text)
    text = _fix_punctuation_spacing(text)
    text = _normalize_whitespace(text)

    return text


if __name__ == "__main__":
    messy = (
        "  The   company\u00a0said it was  NOT  responsible .\n\n\n\n"
        "\u201cWe aren\u2019t happy ,\u201d she said \u2014 profits fell 12.5% "
        "to $3.2\u00a0billion .\t\tShares never recovered\u200b.\n"
    )
    print("BEFORE:\n" + repr(messy))
    print("\nAFTER:\n" + repr(clean_text(messy)))