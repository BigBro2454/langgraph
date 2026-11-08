"""Optional Langfuse tracing for LangGraph runs.

Tracing stays off until both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
set. The same Langfuse project can receive Google ADK traces later. ADK uses
its own OpenTelemetry export, so this callback stays LangGraph-only.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_shutdown_registered = False


def langfuse_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def attach_langfuse(config: dict[str, Any], *, agent_name: str, thread_id: str) -> dict[str, Any]:
    """Return a LangGraph config that also reports the run to Langfuse."""
    if not langfuse_enabled():
        return config

    global _shutdown_registered
    if not _shutdown_registered:
        atexit.register(shutdown_langfuse)
        _shutdown_registered = True

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    get_client()
    handler = CallbackHandler()

    metadata = dict(config.get("metadata") or {})
    metadata.update(
        {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["langgraph", agent_name],
            "framework": "langgraph",
            "agent": agent_name,
        }
    )
    callbacks = list(config.get("callbacks") or [])
    callbacks.append(handler)
    return {
        **config,
        "callbacks": callbacks,
        "run_name": agent_name,
        "metadata": metadata,
    }


def shutdown_langfuse() -> None:
    """Flush queued spans. Short CLI runs exit before the background sender does."""
    if not langfuse_enabled():
        return
    try:
        from langfuse import get_client

        client = get_client()
        client.flush()
        client.shutdown()
    except Exception:
        return
