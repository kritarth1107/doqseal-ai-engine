"""Server-sent events with heartbeats and cancellation on disconnect."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from app.chat.engine import Event

logger = logging.getLogger("doqseal.chat.sse")

_DONE = object()


def format_event(event: Event) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


async def sse_stream(
    events: AsyncIterator[Event],
    *,
    heartbeat_seconds: float,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    """Runs the pipeline in its own task and relays its events as SSE text.

    Sends ": ping" when nothing happened for `heartbeat_seconds`. When the
    consumer stops (client disconnect or cancellation) the pipeline task is
    cancelled, which also closes any open model stream.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()

    async def produce() -> None:
        try:
            async for event in events:
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("chat stream failed")
            await queue.put(
                Event("error", {"code": "internal_error", "message": "Something went wrong. Please try again."})
            )
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
            except TimeoutError:
                if is_disconnected is not None and await is_disconnected():
                    logger.info("chat client disconnected")
                    break
                yield ": ping\n\n"
                continue
            if item is _DONE:
                break
            yield format_event(item)  # type: ignore[arg-type]
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
