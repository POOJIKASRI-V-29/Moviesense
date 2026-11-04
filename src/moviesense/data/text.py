"""Text normalisation, validation and sentence splitting (report §9.2 stage 2, §7.8)."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

from moviesense.config import MAX_INPUT_CHARS

_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f]")
_WS = re.compile(r"\s+")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
_WIKI_MARKUP = re.compile(r"\{\{.*?\}\}|\[\[|\]\]|<[^>]+>")

# Sentence boundary: end punctuation followed by whitespace and an uppercase letter / quote / digit.
# Avoids splitting after common abbreviations.
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "lt", "col", "gen", "capt", "sgt",
           "vs", "etc", "no", "mt", "ft", "u.s", "u.k", "a.m", "p.m", "inc", "co", "rev", "hon"}
_SENT_CANDIDATE = re.compile(r"([.!?][\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")


class InputValidationError(ValueError):
    """Raised when user input is rejected by the validation rules."""


def normalise(text: str) -> str:
    """NFC-normalise, unescape HTML entities, strip control chars/markup, collapse whitespace."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    text = _WIKI_MARKUP.sub(" ", text)
    return _WS.sub(" ", text).strip()


def validate_user_text(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Normalise and validate an inference input. Lower-casing is left to the model."""
    if text is None:
        raise InputValidationError("Input text is required.")
    cleaned = normalise(text)
    if not cleaned:
        raise InputValidationError("Input text is empty.")
    if len(cleaned) > max_chars:
        raise InputValidationError(f"Input text exceeds {max_chars} characters ({len(cleaned)}).")
    if not _ALPHA.search(cleaned):
        raise InputValidationError("Input text contains no alphabetic characters.")
    return cleaned


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (no external model download needed)."""
    sentences: list[str] = []
    start = 0
    for m in _SENT_CANDIDATE.finditer(text):
        end = m.end(1)
        prev_word = re.findall(r"([A-Za-z.]+)$", text[start:m.start(1)])
        if prev_word and prev_word[0].lower().rstrip(".") in _ABBREV:
            continue
        if prev_word and len(prev_word[0]) == 1 and prev_word[0].isupper():  # initials: "J. Smith"
            continue
        sent = text[start:end].strip()
        if sent:
            sentences.append(sent)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def text_hash(text: str) -> str:
    """Hash of aggressively normalised text, for exact-duplicate detection."""
    key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def normalise_title(title: str) -> str:
    """Normalised title key used for split groups (remakes share a group)."""
    t = normalise(title).lower()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return t.strip()
