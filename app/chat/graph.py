"""LangGraph chat skeleton: retrieve → generate → format."""

from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from app.chat.tools import list_document_library, search_documents
from app.config import settings

logger = logging.getLogger("doqseal.chat.graph")

STUB_ANSWER = (
    "I'm running in stub mode because no chat LLM is available. "
    "Configure Azure OpenAI (GPT-5.4) or Ollama to get live answers grounded "
    "in your indexed documents."
)


class ChatState(TypedDict):
    message: str
    organisation_id: str
    project_id: str | None
    user_id: str | None
    context: list[dict[str, Any]]
    library: dict[str, Any]
    answer: str
    citations: list[dict[str, Any]]
    thinking: list[dict[str, str]]
    mode: str


def _intent(message: str) -> str:
    text = (message or "").lower()
    about_rx = bool(re.search(r"prescri|\brx\b", text))
    about_invoice = bool(re.search(r"invoice|cash\s*memo|receipt|\bbills?\b", text))
    counting = bool(re.search(r"how many|how much|\bcount\b|number of|\btotal\b|\bhave\b", text))
    listing = bool(re.search(r"\blist\b|\bshow\b|\bwhich\b|\bwhat are\b|\bname\b", text))
    if about_rx and not about_invoice:
        if listing and not counting:
            return "list_prescriptions"
        return "count_prescriptions"
    if about_invoice and (counting or listing):
        return "count_invoices" if counting or not listing else "list_invoices"
    if re.search(r"\bnotes?\b", text) and (counting or listing):
        return "count_notes"
    if counting and re.search(r"\bdocuments?\b|\bfiles?\b", text):
        return "count_documents"
    return "open"


def _titles(items: list[dict[str, Any]]) -> str:
    if not items:
        return "none"
    return "; ".join(
        (item.get("title") or item.get("filename") or "Untitled") for item in items[:12]
    )


def _md_cell(value: Any) -> str:
    text = str(value or "—").replace("|", "/").replace("\n", " ").strip()
    return text or "—"


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    shown = rows[:40]
    lines = ["| Document | Type | File |", "| --- | --- | --- |"]
    for row in shown:
        kind = str(row.get("kind") or "document").replace("_", " ").title()
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.get("title") or row.get("filename") or "Untitled"),
                    kind,
                    _md_cell(row.get("filename")),
                ]
            )
            + " |"
        )
    table = "\n".join(lines)
    if len(rows) > len(shown):
        table += f"\n\n…and {len(rows) - len(shown)} more."
    return table


def _with_table(summary: str, rows: list[dict[str, Any]]) -> str:
    table = _markdown_table(rows)
    if not table:
        return summary
    return f"{summary}\n\n{table}"


def _inventory_answer(intent: str, library: dict[str, Any]) -> str | None:
    items = list(library.get("items") or [])
    prescriptions = [item for item in items if item.get("kind") == "prescription"]
    invoices = [item for item in items if item.get("kind") == "invoice"]
    notes = [item for item in items if item.get("kind") == "note"]

    if intent == "count_prescriptions":
        n = len(prescriptions)
        noun = "prescription" if n == 1 else "prescriptions"
        if n == 0:
            return (
                f"You have 0 prescriptions. I checked {len(items)} document"
                f"{'' if len(items) == 1 else 's'} in Drive and none are prescriptions."
            )
        return _with_table(f"You have {n} {noun} in Drive.", prescriptions)
    if intent == "list_prescriptions":
        if not prescriptions:
            return "I didn't find any prescriptions in the documents you can see."
        return _with_table("These are the prescriptions in Drive.", prescriptions)
    if intent == "count_invoices":
        n = len(invoices)
        noun = "invoice" if n == 1 else "invoices"
        return _with_table(f"You have {n} {noun} in Drive.", invoices)
    if intent == "list_invoices":
        if not invoices:
            return "I didn't find any invoices in the documents you can see."
        return _with_table("These are the invoices in Drive.", invoices)
    if intent == "count_notes":
        n = len(notes)
        noun = "note" if n == 1 else "notes"
        return _with_table(f"You have {n} {noun} in Drive.", notes)
    if intent == "count_documents":
        summary = (
            f"You have {len(items)} documents in Drive "
            f"({library.get('prescriptionCount', 0)} prescriptions, "
            f"{library.get('invoiceCount', 0)} invoices, "
            f"{library.get('noteCount', 0)} notes, "
            f"{library.get('otherCount', 0)} other)."
        )
        return _with_table(summary, items)
    return None


def _thinking_for(message: str, intent: str, library: dict[str, Any]) -> list[dict[str, str]]:
    items = list(library.get("items") or [])
    prescriptions = [item for item in items if item.get("kind") == "prescription"]
    invoices = [item for item in items if item.get("kind") == "invoice"]
    notes = [item for item in items if item.get("kind") == "note"]
    steps = [
        {
            "title": "Read the question",
            "detail": _clip_msg(message, 180),
        },
        {
            "title": "Scan Drive",
            "detail": f"Loaded {len(items)} visible document{'s' if len(items) != 1 else ''} from the library.",
        },
        {
            "title": "Classify each file",
            "detail": (
                f"{len(prescriptions)} prescriptions, {len(invoices)} invoices, "
                f"{len(notes)} notes, {library.get('otherCount', 0)} other."
            ),
        },
    ]
    if intent.startswith("count_prescription") or intent.startswith("list_prescription"):
        steps.append(
            {
                "title": "Match prescriptions only",
                "detail": _titles(prescriptions) if prescriptions else "No prescription files in Drive.",
            }
        )
    elif "invoice" in intent:
        steps.append(
            {
                "title": "Match invoices only",
                "detail": _titles(invoices),
            }
        )
    elif intent == "count_notes":
        steps.append(
            {
                "title": "Match notes only",
                "detail": _titles(notes),
            }
        )
    elif intent == "count_documents":
        steps.append(
            {
                "title": "List every file",
                "detail": f"{len(items)} document{'s' if len(items) != 1 else ''} in Drive.",
            }
        )
    else:
        steps.append(
            {
                "title": "Pull supporting excerpts",
                "detail": "Using the classified library plus the closest indexed passages.",
            }
        )
    steps.append(
        {
            "title": "Write the answer",
            "detail": "Counts come from the classified Drive list, not a guess.",
        }
    )
    return steps


def _library_block(library: dict[str, Any] | None) -> str:
    library = library or {}
    items = library.get("items") or []
    if not items:
        return (
            "Document library: the user has no visible documents in Drive "
            "for this scope."
        )

    lines = []
    for item in items[:40]:
        kind = item.get("kind") or ("prescription" if item.get("prescription") else "document")
        lines.append(
            f"- [{kind}] {item.get('title') or 'Untitled'} "
            f"(id={item.get('documentId') or '?'}, file={item.get('filename') or '?'})"
        )
    more = ""
    if len(items) > 40:
        more = f"\n…and {len(items) - 40} more not listed."
    return (
        "Document library (authoritative inventory from the user's Drive):\n"
        f"- Total documents: {library.get('total', len(items))}\n"
        f"- Prescriptions: {library.get('prescriptionCount', 0)}\n"
        + "\n".join(lines)
        + more
        + "\nWhen the user asks how many prescriptions (or documents) they have, "
        "use these counts exactly. Invoices, cash memos, and notes are not prescriptions.\n"
    )


def _build_prompt(
    message: str,
    context: list[dict[str, Any]],
    library: dict[str, Any] | None,
) -> str:
    context_block = ""
    if context:
        context_block = "\n\nRetrieved excerpts:\n" + "\n\n".join(
            f"[{idx + 1}] {chunk.get('documentId', '?')}: "
            f"{_clip_msg(str(chunk.get('snippet', '')), 400)}"
            for idx, chunk in enumerate(context[:4])
        )

    return (
        "You are DoqSeal intelligence. Answer from the document library and excerpts. "
        "Be brief and specific. Cite document titles when useful.\n\n"
        f"{_library_block(library)}"
        f"{context_block}\n\n"
        f"User: {_clip_msg(message)}\nAssistant:"
    )


def _clip_msg(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _call_azure_openai_chat(prompt: str) -> str | None:
    endpoint = (settings.azure_openai_endpoint or "").rstrip("/")
    key = settings.azure_openai_api_key or ""
    deployment = (
        settings.azure_openai_text_deployment
        or settings.azure_openai_deployment
        or "gpt-4.1-mini"
    )
    if not endpoint or not key:
        return None
    url = (
        f"{endpoint}/openai/deployments/{deployment}/chat/completions"
        f"?api-version={settings.azure_openai_api_version}"
    )
    payload = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max(settings.chat_max_completion_tokens, 500),
    }
    try:
        with httpx.Client(timeout=45.0) as client:
            response = client.post(
                url,
                headers={"api-key": key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            text = (
                ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
                or ""
            ).strip()
            return text or None
    except Exception as exc:
        logger.warning("Azure OpenAI chat failed: %s", exc)
        return None


def _call_ollama(prompt: str) -> str | None:
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.llm_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=90.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            body = response.json()
            text = (body.get("response") or "").strip()
            return text or None
    except Exception as exc:
        logger.warning("Ollama call failed (%s): %s", settings.ollama_url, exc)
        return None


def retrieve_node(state: ChatState) -> dict[str, Any]:
    library: dict[str, Any] = {
        "total": 0,
        "prescriptionCount": 0,
        "items": [],
    }
    try:
        library = list_document_library(
            state["organisation_id"],
            project_id=state.get("project_id"),
            user_id=state.get("user_id"),
        )
    except Exception as exc:
        logger.warning("Document library lookup failed: %s", exc)

    chunks = search_documents(
        state["organisation_id"],
        state["message"],
        project_id=state.get("project_id"),
        user_id=state.get("user_id"),
    )
    return {"context": chunks, "library": library}


def generate_node(state: ChatState) -> dict[str, Any]:
    library = state.get("library") or {}
    intent = _intent(state["message"])
    thinking = _thinking_for(state["message"], intent, library)
    direct = _inventory_answer(intent, library)
    if direct:
        return {"answer": direct, "mode": "library", "thinking": thinking}

    prompt = _build_prompt(
        state["message"],
        state.get("context") or [],
        library,
    )
    answer = _call_azure_openai_chat(prompt)
    if answer:
        return {"answer": answer, "mode": "live", "thinking": thinking}
    answer = _call_ollama(prompt)
    if answer:
        return {"answer": answer, "mode": "live", "thinking": thinking}
    return {"answer": STUB_ANSWER, "mode": "stub", "thinking": thinking}


def format_node(state: ChatState) -> dict[str, Any]:
    intent = _intent(state["message"])
    library = state.get("library") or {}
    items = list(library.get("items") or [])
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(item: dict[str, Any]) -> None:
        document_id = item.get("documentId")
        if not document_id or document_id in seen:
            return
        seen.add(document_id)
        title = item.get("title") or item.get("filename") or "Document"
        kind = item.get("kind") or "document"
        citations.append(
            {
                "documentId": document_id,
                "projectId": item.get("projectId"),
                "title": title,
                "kind": kind,
                "filename": item.get("filename") or "",
                "snippet": f"{kind.replace('_', ' ').title()}: {title}",
            }
        )

    if intent in {"count_prescriptions", "list_prescriptions"}:
        matched = [item for item in items if item.get("kind") == "prescription"]
    elif intent in {"count_invoices", "list_invoices"}:
        matched = [item for item in items if item.get("kind") == "invoice"]
    elif intent == "count_notes":
        matched = [item for item in items if item.get("kind") == "note"]
    elif intent == "count_documents":
        matched = items[:40]
    else:
        matched = []
        by_id = {item.get("documentId"): item for item in items}
        for chunk in state.get("context") or []:
            known = by_id.get(chunk.get("documentId"))
            if known:
                _add(known)
            elif chunk.get("documentId"):
                _add(
                    {
                        "documentId": chunk.get("documentId"),
                        "projectId": chunk.get("projectId"),
                        "title": (chunk.get("snippet") or "Document")[:80],
                        "kind": "document",
                    }
                )
        return {"citations": citations[:8]}

    for item in matched[:40]:
        _add(item)
    return {"citations": citations}


def _build_graph():
    workflow = StateGraph(ChatState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("format", format_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "format")
    workflow.add_edge("format", END)
    return workflow.compile()


_graph = _build_graph()


def run_chat(
    message: str,
    organisation_id: str,
    *,
    project_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    result = _graph.invoke(
        {
            "message": message,
            "organisation_id": organisation_id,
            "project_id": project_id,
            "user_id": user_id,
            "context": [],
            "library": {"total": 0, "prescriptionCount": 0, "items": []},
            "answer": "",
            "citations": [],
            "thinking": [],
            "mode": "stub",
        }
    )
    return {
        "answer": result.get("answer", STUB_ANSWER),
        "citations": result.get("citations") or [],
        "thinking": result.get("thinking") or [],
        "mode": result.get("mode", "stub"),
    }
