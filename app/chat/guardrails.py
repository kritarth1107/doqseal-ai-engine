"""Chat guardrails: relevance gate, grounding, citation verification, injection defense."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.config import settings
from app.db.mongo import get_org_config

logger = logging.getLogger("doqseal.chat.guardrails")

SMALL_TALK_PATTERNS = [
    r"^\s*(hi|hello|hey(\s+there)?|good\s*(morning|afternoon|evening)|howdy)\s*[!.?]*\s*$",
    r"^\s*(thanks?|thank\s*you|thx)\s*[!.?]*\s*$",
    r"^\s*(bye|goodbye|see\s*you|later)\s*[!.?]*\s*$",
    r"^\s*(what\s*can\s*you\s*do|help|what\s*are\s*you|who\s*are\s*you)\s*[?!.]*\s*$",
    r"^\s*(how\s*are\s*you|how\s*do\s*you\s*do)\s*[?!.]*\s*$",
]

SMALL_TALK_RESPONSE = (
    "Hello! I'm DoqSeal Intelligence. I can help you find information in your "
    "organisation's documents, answer questions about their contents, and provide "
    "summaries with citations. What would you like to know about your documents?"
)

DECLINE_TEMPLATES = {
    "not_covered": (
        "I can only answer questions using documents in your organisation's library. "
        "I couldn't find relevant information about this in your documents. "
        "Would you like to upload a document that might help, or try rephrasing your question?"
    ),
    "off_topic": (
        "I'm designed to help you find information in your organisation's documents. "
        "This question seems to be outside that scope. "
        "Is there something about your documents I can help you with?"
    ),
    "small_talk": SMALL_TALK_RESPONSE,
    "fabricated_citation": (
        "I wasn't able to provide a properly sourced answer to your question. "
        "The information I found may not fully support a complete response. "
        "Could you try asking about a more specific aspect of your documents?"
    ),
}

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(all\s+)?previous",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are\s+)?a",
    r"pretend\s+(to\s+be|you\s+are)",
    r"system\s*:\s*",
    r"<\s*/?system\s*>",
    r"</?\s*document\s*>",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"show\s+(me\s+)?(your\s+)?(system\s+)?instructions?",
    r"what\s+(are\s+)?(your\s+)?instructions?",
    r"doqseal\s+(should\s+)?(approve|reject)",
]

BANNED_OUTPUT_PATTERNS = [
    r"\bapprove[sd]?\b",
    r"\breject(ed|s)?\b",
    r"\b\d+%\s*(accuracy|correct|precise)",
    r"\bcertified\b",
]


@dataclass
class GuardrailConfig:
    """Guardrail thresholds - can be overridden per organisation."""

    min_rerank_score: float = 0.25
    min_chunks: int = 1
    coverage_threshold: float = 0.6

    @classmethod
    def for_org(cls, organisation_id: str) -> GuardrailConfig:
        """Load config with per-org overrides."""
        config = cls(
            min_rerank_score=settings.guardrail_min_rerank_score,
            min_chunks=settings.guardrail_min_chunks,
            coverage_threshold=settings.guardrail_coverage_threshold,
        )

        org_config = get_org_config(organisation_id)
        if org_config:
            if "minRerankScore" in org_config:
                config.min_rerank_score = float(org_config["minRerankScore"])
            if "minChunks" in org_config:
                config.min_chunks = int(org_config["minChunks"])
            if "coverageThreshold" in org_config:
                config.coverage_threshold = float(org_config["coverageThreshold"])

        return config


@dataclass
class GuardrailResult:
    """Result of guardrail checks."""

    passed: bool
    reason: str | None = None
    decline_type: str | None = None
    decline_message: str | None = None
    sanitized_query: str | None = None
    supporting_chunks: list[dict[str, Any]] = field(default_factory=list)
    injection_detected: bool = False


def is_small_talk(message: str) -> bool:
    """Check if message is small talk that gets a canned response."""
    text = message.strip().lower()
    for pattern in SMALL_TALK_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            return True
    return False


def detect_injection(text: str) -> bool:
    """Detect potential prompt injection attempts."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning("Injection pattern detected: %s", pattern)
            return True
    return False


def sanitize_query(message: str) -> str:
    """Sanitize user query to neutralize injection attempts."""
    sanitized = message

    for pattern in INJECTION_PATTERNS:
        sanitized = re.sub(pattern, "[filtered]", sanitized, flags=re.IGNORECASE)

    sanitized = re.sub(r"<[^>]+>", "", sanitized)

    if len(sanitized) > 8000:
        sanitized = sanitized[:8000]

    return sanitized.strip()


def sanitize_document_content(text: str) -> str:
    """Sanitize document content before including in prompts."""
    if not text:
        return ""

    sanitized = text

    sanitized = re.sub(r"</?\s*document\s*[^>]*>", "[doc-boundary]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"</?\s*system\s*[^>]*>", "[sys-boundary]", sanitized, flags=re.IGNORECASE)

    sanitized = re.sub(r"(system|user|assistant)\s*:\s*", r"\1 - ", sanitized, flags=re.IGNORECASE)

    if len(sanitized) > 50000:
        sanitized = sanitized[:50000] + "..."

    return sanitized


def check_banned_output(text: str) -> list[str]:
    """Check if output contains banned phrases."""
    violations = []
    for pattern in BANNED_OUTPUT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(pattern)
    return violations


def score_gate(
    chunks: list[dict[str, Any]],
    config: GuardrailConfig,
) -> tuple[bool, list[dict[str, Any]]]:
    """Apply score-based relevance gate.

    Returns (passed, filtered_chunks).
    """
    if not chunks:
        return False, []

    passing_chunks = [c for c in chunks if c.get("score", 0) >= config.min_rerank_score]

    if len(passing_chunks) < config.min_chunks:
        return False, []

    return True, passing_chunks


async def check_coverage(
    query: str,
    chunks: list[dict[str, Any]],
    config: GuardrailConfig,
) -> tuple[bool, list[dict[str, Any]], str | None]:
    """Check if chunks cover the query using LLM.

    Returns (covered, supporting_chunks, reason).
    """
    if not chunks:
        return False, [], "No relevant documents found"

    context_text = "\n\n".join(
        f"[{i + 1}] {sanitize_document_content(c.get('text', '')[:1500])}"
        for i, c in enumerate(chunks[:8])
    )

    prompt = f"""Analyze whether the following document excerpts contain information to answer the question.

Question: {sanitize_query(query)}

Document excerpts:
{context_text}

Respond with JSON only:
{{"covered": true/false, "supporting_ids": [1, 2, ...], "reason": "brief explanation"}}

Rules:
- "covered" is true ONLY if the excerpts contain specific information to answer the question
- "supporting_ids" lists the excerpt numbers that contain relevant information
- Do not use outside knowledge - only what's in the excerpts"""

    try:
        result = await _call_coverage_check(prompt)
        if result:
            covered = result.get("covered", False)
            supporting_ids = result.get("supporting_ids", [])
            reason = result.get("reason")

            supporting_chunks = [chunks[i - 1] for i in supporting_ids if 0 < i <= len(chunks)]

            return covered, supporting_chunks, reason
    except Exception as exc:
        logger.warning("Coverage check failed: %s", exc)

    return len(chunks) >= config.min_chunks, chunks[: config.min_chunks], None


async def _call_coverage_check(prompt: str) -> dict[str, Any] | None:
    """Call LLM for coverage check."""
    endpoint = (settings.azure_openai_endpoint or "").rstrip("/")
    key = settings.azure_openai_api_key or ""
    deployment = settings.azure_openai_text_deployment or "gpt-4.1-mini"

    if not endpoint or not key:
        return None

    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={settings.azure_openai_api_version}"
    )

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            url,
            headers={"api-key": key, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")

        import json

        return json.loads(content)


@dataclass
class Citation:
    """A citation reference."""

    n: int
    document_id: str
    title: str
    page: int | None
    quote: str


def verify_citations(
    answer: str,
    citations: list[Citation],
    chunks: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    """Verify that citations are grounded in the source chunks.

    Returns (all_valid, list_of_errors).
    """
    errors = []

    citation_pattern = re.compile(r"\[(\d+)\]")
    used_citations = {int(m) for m in citation_pattern.findall(answer)}

    chunk_texts = {c.get("documentId", ""): c.get("text", "").lower() for c in chunks}

    citation_map = {c.n: c for c in citations}

    for n in used_citations:
        if n not in citation_map:
            errors.append(f"Citation [{n}] referenced but not provided")
            continue

        citation = citation_map[n]
        doc_text = chunk_texts.get(citation.document_id, "")

        if not doc_text:
            errors.append(f"Citation [{n}] references unknown document")
            continue

        quote_words = set(citation.quote.lower().split())
        doc_words = set(doc_text.split())
        overlap = len(quote_words & doc_words) / max(len(quote_words), 1)

        if overlap < 0.5:
            errors.append(f"Citation [{n}] quote not found in document")

    return len(errors) == 0, errors


def apply_guardrails(
    message: str,
    organisation_id: str,
) -> GuardrailResult:
    """Apply all pre-generation guardrails.

    Returns result indicating whether to proceed with generation.
    """
    if is_small_talk(message):
        return GuardrailResult(
            passed=False,
            reason="small_talk",
            decline_type="small_talk",
            decline_message=DECLINE_TEMPLATES["small_talk"],
        )

    injection = detect_injection(message)
    sanitized = sanitize_query(message)

    return GuardrailResult(
        passed=True,
        sanitized_query=sanitized,
        injection_detected=injection,
    )


async def apply_relevance_gate(
    query: str,
    chunks: list[dict[str, Any]],
    organisation_id: str,
) -> GuardrailResult:
    """Apply relevance gate after retrieval.

    Checks score threshold and coverage.
    """
    config = GuardrailConfig.for_org(organisation_id)

    passed_score, filtered_chunks = score_gate(chunks, config)

    if not passed_score:
        return GuardrailResult(
            passed=False,
            reason="not_covered",
            decline_type="not_covered",
            decline_message=DECLINE_TEMPLATES["not_covered"],
        )

    covered, supporting, reason = await check_coverage(query, filtered_chunks, config)

    if not covered:
        return GuardrailResult(
            passed=False,
            reason=reason or "not_covered",
            decline_type="not_covered",
            decline_message=DECLINE_TEMPLATES["not_covered"],
        )

    return GuardrailResult(
        passed=True,
        supporting_chunks=supporting,
    )


def get_decline_message(decline_type: str) -> str:
    """Get the decline message for a given type."""
    return DECLINE_TEMPLATES.get(decline_type, DECLINE_TEMPLATES["not_covered"])
