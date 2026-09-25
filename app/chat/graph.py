"""LangGraph chat skeleton: retrieve → generate → format."""

from __future__ import annotations

import logging
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
    mode: str


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
        kind = "prescription" if item.get("prescription") else "document"
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
        "use these counts. Do not say you cannot access their files."
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
    prompt = _build_prompt(
        state["message"],
        state.get("context") or [],
        state.get("library"),
    )
    # Prefer Azure OpenAI for chat; fall back to Ollama if configured.
    answer = _call_azure_openai_chat(prompt)
    if answer:
        return {"answer": answer, "mode": "live"}
    answer = _call_ollama(prompt)
    if answer:
        return {"answer": answer, "mode": "live"}
    return {"answer": STUB_ANSWER, "mode": "stub"}


def format_node(state: ChatState) -> dict[str, Any]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(document_id: str | None, project_id: str | None, snippet: str) -> None:
        if not document_id or document_id in seen:
            return
        seen.add(document_id)
        citations.append(
            {
                "documentId": document_id,
                "projectId": project_id,
                "snippet": snippet,
            }
        )

    for chunk in state.get("context") or []:
        _add(chunk.get("documentId"), chunk.get("projectId"), chunk.get("snippet", ""))

    library = state.get("library") or {}
    items = list(library.get("items") or [])
    # Prefer prescription matches so "how many prescriptions" surfaces those files
    items.sort(key=lambda item: 0 if item.get("prescription") else 1)
    for item in items:
        if len(citations) >= 8:
            break
        title = item.get("title") or item.get("filename") or "Document"
        kind = "Prescription" if item.get("prescription") else "Document"
        _add(item.get("documentId"), item.get("projectId"), f"{kind}: {title}")

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
            "mode": "stub",
        }
    )
    return {
        "answer": result.get("answer", STUB_ANSWER),
        "citations": result.get("citations") or [],
        "mode": result.get("mode", "stub"),
    }
