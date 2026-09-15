"""
persistent_agent — Stateful LangGraph Agent with Checkpointing and Time Travel.

Demonstrates:
- MemorySaver checkpointer for thread-isolated persistent memory
- Multi-turn conversation across distinct thread_ids
- Custom state variables beyond messages (e.g., user_name, session_notes)
- State inspection via graph.get_state(config)
- Time travel / state mutation via graph.update_state(config, values)
"""

import operator
from typing import Annotated, List, Optional, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from shared.utils.model_factory import get_model


class PersistentState(TypedDict):
    """Extended state schema with conversation history and persistent scratchpad."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_name: Optional[str]
    notes: Annotated[List[str], operator.add]


# Initialize model
raw_model = get_model()

SYSTEM_INSTRUCTION = (
    "You are an intelligent personal memory assistant. "
    "You remember user details across turns and help organize notes and preferences. "
    "Be concise, attentive, and proactive."
)


def assistant_node(state: PersistentState) -> dict:
    """Processes messages and updates memory state."""
    messages = list(state.get("messages", []))
    user_name = state.get("user_name")
    notes = state.get("notes", [])

    context_prompt = SYSTEM_INSTRUCTION
    if user_name:
        context_prompt += f"\nUser's confirmed name: {user_name}"
    if notes:
        context_prompt += f"\nActive Notes on file: {', '.join(notes)}"

    full_messages = [SystemMessage(content=context_prompt)] + messages
    response = raw_model.invoke(full_messages)

    # Simple heuristic to extract user name if stated
    updates = {"messages": [response]}
    last_user_msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_msg = m.content.lower()
            break

    if "my name is " in last_user_msg and not user_name:
        name_extracted = last_user_msg.split("my name is ")[-1].split()[0].title()
        updates["user_name"] = name_extracted
        updates["notes"] = [f"User identified as {name_extracted}"]

    return updates


# Build workflow
builder = StateGraph(PersistentState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# In-memory checkpointer enables persistence per thread_id
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
