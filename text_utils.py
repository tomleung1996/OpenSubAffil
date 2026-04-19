"""Text normalisation utilities shared across pipeline steps."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from config import (
    DEPARTMENT_CUE_TOKENS,
    PREFIX_EXCLUSION_TOKENS,
    STOP_WORDS,
    TYPE_WORDS,
    TYPE_WORDS_ABBR,
)

_WHITESPACE_RE = re.compile(r"\s+")
_ABBR_RE = re.compile("|".join(r"\b" + re.escape(abbr) + r"\b" for abbr in TYPE_WORDS_ABBR))
_TYPE_WORD_RE = re.compile(r"\b(%s)\b" % "|".join(re.escape(w) for w in TYPE_WORDS), re.IGNORECASE)
_STOP_WORD_RE = re.compile(r"\b(%s)\b" % "|".join(re.escape(w) for w in STOP_WORDS), re.IGNORECASE)


def clean_affiliation_string(text: Any) -> str:
    """Normalise a raw affiliation string before feeding it to the NER models.

    Collapses whitespace, repairs punctuation spacing, and strips wrapping
    brackets. Case is preserved because the NER model is case-sensitive.
    """
    if not isinstance(text, str):
        return ""
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r"\s+([,;:.)\]}])", r"\1", text)
    text = re.sub(r"([,;:.({\[])\s+", r"\1", text)
    text = re.sub(r"([,;:.)\]}])([a-zA-Z0-9])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z0-9])([({\[])", r"\1 \2", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"^[\(\[\{]+", "", text)
    text = re.sub(r"[\)\]\}]+$", "", text)
    return text


def clean_department_string(raw_str: Any) -> str:
    """Normalise a department string extracted by the NER model.

    Returns an empty string when the input does not contain any type word,
    because such strings are typically noise and cannot be disambiguated.
    """
    if not raw_str:
        return ""
    text = str(raw_str).lower()
    text = re.sub(r"\(.*$", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = text.replace("amp", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()

    words = text.split()
    while words and words[0] in STOP_WORDS:
        words.pop(0)
    while words and words[-1] in STOP_WORDS:
        words.pop()
    text = " ".join(words)

    if not any(w in text for w in TYPE_WORDS):
        return ""

    text = re.sub(r"^\d+\s*", "", text)
    return text.strip()


def is_abbreviation(dept_str: str) -> bool:
    """Return True if the string contains at least one type-word abbreviation."""
    return bool(_ABBR_RE.search(dept_str))


def preprocess_department_name(name: Any) -> str:
    """Aggressive normalisation used by the deduplication pre-processing.

    Unlike `clean_department_string` this drops diacritics and keeps names
    that have no type word (so that downstream core-name extraction can
    still operate on them).
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name or "")
    text = text.encode("ascii", "ignore").decode("ascii", "ignore").lower()
    text = text.replace("&", " and ").replace(".", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_core_name(full_name: str) -> str:
    """Return the tokens that follow the first type word, with stop words removed.

    Falls back to the tokens before the first type word, and finally to the
    normalised name itself if no type word is present.
    """
    normalized = preprocess_department_name(full_name)
    if not normalized:
        return ""
    match = _TYPE_WORD_RE.search(normalized)
    if not match:
        core = normalized
    else:
        before = normalized[: match.start()].strip()
        after = normalized[match.end():].strip()
        core = after or before
    core = _STOP_WORD_RE.sub("", core)
    core = _WHITESPACE_RE.sub(" ", core).strip()
    return core or normalized


def _clean_prefix_token(token: str) -> str:
    return token.replace("-", "").replace("'", "").replace(".", "")


def _is_valid_prefix_token(token: str) -> bool:
    cleaned = _clean_prefix_token(token)
    if not cleaned or not cleaned.isalpha():
        return False
    return cleaned not in PREFIX_EXCLUSION_TOKENS


def split_prefix_suffix(
    dept_name: str,
    cue_tokens: tuple[str, ...] = DEPARTMENT_CUE_TOKENS,
    max_prefix_tokens: int = 3,
) -> tuple[str, str, bool]:
    """Split a department name at the first cue token.

    Returns (prefix, suffix, prefix_is_person_like). The suffix includes the
    cue token itself. An empty triple means no cue token was found or the
    split was rejected (e.g. the suffix would be a single token).
    """
    tokens = dept_name.split()
    for idx, token in enumerate(tokens):
        if token not in cue_tokens:
            continue
        prefix_tokens = tokens[:idx]
        suffix_tokens = tokens[idx:]
        if len(suffix_tokens) <= 1:
            continue
        prefix = " ".join(prefix_tokens).strip()
        suffix = " ".join(suffix_tokens).strip()
        has_prefix = bool(prefix_tokens)
        prefix_valid = (
            has_prefix
            and len(prefix_tokens) <= max_prefix_tokens
            and all(_is_valid_prefix_token(tok) for tok in prefix_tokens)
        )
        return prefix, suffix, prefix_valid
    return "", "", False
