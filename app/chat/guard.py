"""Guardrails for grounded chat: scope replies, sanitising, output wording, citations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.chat.retrieval import Evidence, query_terms

SCOPE_REPLY = (
    "Hi! I answer questions using the documents your organisation has uploaded to "
    "DoqSeal, and I cite the documents I use. Ask me about something in your documents, "
    "for example a name, an amount, a date or a summary of a file."
)

DECLINE_MESSAGES = {
    "not_covered": (
        "I can only answer from documents uploaded to your organisation, and I couldn't "
        "find anything in them that covers this. If you have a document about it, upload "
        "it and ask again."
    ),
    "off_topic": (
        "That's outside what I can help with. I answer questions using your " "organisation's documents only."
    ),
    "small_talk": SCOPE_REPLY,
}

_SMALL_TALK = [
    r"(hi|hello|hey|hiya|namaste|good\s*(morning|afternoon|evening))( there)?",
    r"(thanks?|thank\s*you|thx|ok(ay)?|cool|great|nice)( a lot| so much)?",
    r"(bye|goodbye|see\s*you|cya)",
    r"(how\s*are\s*you|how's\s*it\s*going)",
    r"(who|what)\s*are\s*you",
    r"what\s*can\s*you\s*do",
    r"help",
]
_SMALL_TALK_RE = re.compile(r"^\s*(" + "|".join(_SMALL_TALK) + r")\s*[!.?,]*\s*(doqseal)?\s*[!.?]*\s*$", re.IGNORECASE)

_LIBRARY_WORDS = frozenset(
    "document documents doc docs file files upload uploads uploaded many count number total "
    "library drive have has we our my there list show which what did do i me all are is".split()
)
_LIBRARY_RE = re.compile(r"\b(how many|list|show|which|what)\b.*\b(documents?|files?|uploads?|docs?)\b", re.IGNORECASE)


def is_small_talk(message: str) -> bool:
    return bool(_SMALL_TALK_RE.match(message.strip()))


def is_library_question(message: str) -> bool:
    """Questions about the library itself, e.g. "how many documents do we have?"."""
    if not _LIBRARY_RE.search(message):
        return False
    return all(t in _LIBRARY_WORDS for t in query_terms(message))


_TAG_RE = re.compile(r"<\s*/?\s*(document|system|assistant|user|developer|instructions?)\b[^>]*>", re.I)
_ROLE_RE = re.compile(r"^\s*(system|assistant|developer)\s*:", re.I | re.M)


def sanitize_document_text(text: str, limit: int = 12000) -> str:
    """Neutralise markup that could fake a prompt boundary; document text stays data."""
    cleaned = _TAG_RE.sub("[tag removed]", text or "")
    cleaned = _ROLE_RE.sub(lambda m: f"{m.group(1)} -", cleaned)
    cleaned = re.sub(r"(.)\1{40,}", lambda m: m.group(1) * 40, cleaned)
    return cleaned[:limit]


def sanitize_attr(value: str) -> str:
    return re.sub(r'["<>\n\r]', " ", value or "")[:200]


# --- Output wording --------------------------------------------------------

_BANNED_RE = re.compile(r"\b(approv|reject)(\w*)", re.IGNORECASE)
_REPLACEMENTS = {
    ("approv", "e"): "accept",
    ("approv", "es"): "accepts",
    ("approv", "ed"): "accepted",
    ("approv", "ing"): "accepting",
    ("approv", "al"): "acceptance",
    ("approv", "als"): "acceptances",
    ("reject", ""): "decline",
    ("reject", "s"): "declines",
    ("reject", "ed"): "declined",
    ("reject", "ing"): "declining",
    ("reject", "ion"): "decline",
    ("reject", "ions"): "declines",
}


def _replace_banned(match: re.Match[str]) -> str:
    stem, suffix = match.group(1), match.group(2)
    word = _REPLACEMENTS.get((stem.lower(), suffix.lower()))
    if word is None:
        word = "accept" if stem.lower() == "approv" else "decline"
    if stem[0].isupper():
        word = word[0].upper() + word[1:]
    if stem.isupper() and len(stem) > 1:
        word = word.upper()
    return word


def clean_wording(text: str) -> str:
    return _BANNED_RE.sub(_replace_banned, text)


class StreamingWordFilter:
    """Applies clean_wording to a token stream without splitting words."""

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, text: str) -> str:
        self._pending += text
        match = re.search(r"\w+$", self._pending)
        cut = match.start() if match else len(self._pending)
        ready, self._pending = self._pending[:cut], self._pending[cut:]
        return clean_wording(ready)

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return clean_wording(ready)


# --- Citations ---------------------------------------------------------------

_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


@dataclass
class Citation:
    n: int
    document_id: str
    title: str
    page: int | None
    quote: str


@dataclass
class Verification:
    ok: bool
    citations: list[Citation]
    errors: list[str]


def cited_numbers(answer: str) -> list[int]:
    seen: list[int] = []
    for match in _CITE_RE.finditer(answer):
        for part in match.group(1).split(","):
            n = int(part.strip())
            if n not in seen:
                seen.append(n)
    return seen


def _best_quote(evidence_text: str, answer_sentences: list[str]) -> str:
    target = set(query_terms(" ".join(answer_sentences)))
    best, best_score = "", -1.0
    for raw in _SENTENCE_RE.findall(evidence_text):
        sentence = raw.strip()
        if len(sentence) < 8:
            continue
        terms = set(query_terms(sentence))
        score = len(terms & target) / (len(terms) ** 0.5 or 1.0)
        if score > best_score:
            best, best_score = sentence, score
    if not best:
        best = evidence_text.strip()
    return best[:240]


def verify_and_cite(answer: str, evidence: list[Evidence]) -> Verification:
    """Every [n] must point at supplied evidence and the answer must cite something.

    Quotes are taken verbatim from the cited evidence, so they always appear in it.
    """
    numbers = cited_numbers(answer)
    errors = [f"[{n}] does not match any document" for n in numbers if n < 1 or n > len(evidence)]
    valid = [n for n in numbers if 1 <= n <= len(evidence)]
    if not valid:
        errors.append("the answer cites no document")
    if errors:
        return Verification(ok=False, citations=[], errors=errors)

    sentences = [s for s in _SENTENCE_RE.findall(answer)]
    citations: list[Citation] = []
    for n in valid:
        item = evidence[n - 1]
        citing = [s for s in sentences if re.search(rf"\[(?:[\d\s,]*\b)?{n}\b", s)]
        citations.append(
            Citation(
                n=n,
                document_id=item.document_id,
                title=item.title,
                page=item.page,
                quote=_best_quote(item.text, citing or sentences),
            )
        )
    return Verification(ok=True, citations=citations, errors=[])
