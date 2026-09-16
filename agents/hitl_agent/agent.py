"""
hitl_agent — Human-in-the-Loop Approval Agent.

Demonstrates:
- Dynamic interrupt() primitive for runtime human approval
- Pausing execution at safety-critical boundaries (e.g. fund transfers, deleting data)
- Resuming workflows using Command(resume=value)
- Persistent checkpointing maintaining state while waiting for human intervention
"""

from typing import Annotated, Sequence, TypedDict, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from shared.utils.model_factory import get_model


class HITLState(TypedDict):
    """State schema for human-in-the-loop agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    action_pending: Optional[str]
    approval_status: Optional[str]
    execution_result: Optional[str]


model = get_model()

SYSTEM_INSTRUCTION = (
    "You are an Operations Executive Agent. Users can ask you to run operations like "
    "'transfer $500 to vendor', 'deploy release v2', or 'delete cache'. "
    "When a sensitive operational request is made, formulate the exact action proposal."
)


def propose_action_node(state: HITLState) -> dict:
    """Evaluates the user request and flags high-impact operations for approval."""
    messages = list(state.get("messages", []))
    last_msg = messages[-1].content if messages else ""

    # Check for sensitive action keywords
    sensitive_keywords = ["transfer", "delete", "deploy", "reset", "purchase"]
    is_sensitive = any(kw in last_msg.lower() for kw in sensitive_keywords)

    if is_sensitive:
        action_desc = f"CONFIRMATION_REQUIRED: Request to perform '{last_msg}'"
        return {
            "action_pending": action_desc,
            "messages": [
                HumanMessage(content=f"[System Warning] Sensitive operation requested: '{last_msg}'. Initiating approval gate.")
            ]
        }
    else:
        response = model.invoke([SystemMessage(content=SYSTEM_INSTRUCTION)] + messages)
        return {"messages": [response], "action_pending": None}


def human_gate_node(state: HITLState) -> dict:
    """Interrupts graph execution and awaits human confirmation."""
    pending = state.get("action_pending")
    if not pending:
        return {}

    # Dynamic interrupt: halts execution, stores state in checkpointer,
    # and surfaces this prompt to the client.
    approval_decision = interrupt({
        "question": f"Approval needed for action: {pending}",
        "options": ["APPROVED", "REJECTED"],
        "action": pending,
    })

    return {"approval_status": str(approval_decision).upper()}


def execute_action_node(state: HITLState) -> dict:
    """Executes or aborts the operation based on the human's decision."""
    status = state.get("approval_status")
    action = state.get("action_pending")

    if status == "APPROVED":
        result = f"SUCCESS: Action '{action}' has been authorized and securely executed."
    else:
        result = f"ABORTED: Action '{action}' was rejected by human operator."

    return {
        "execution_result": result,
        "action_pending": None,
        "messages": [HumanMessage(content=f"[Execution Gate] {result}")],
    }


def should_execute(state: HITLState) -> str:
    """Routes based on whether sensitive approval is needed."""
    if state.get("action_pending"):
        return "human_gate"
    return END


# Build workflow
builder = StateGraph(HITLState)

builder.add_node("propose", propose_action_node)
builder.add_node("human_gate", human_gate_node)
builder.add_node("execute", execute_action_node)

builder.add_edge(START, "propose")
builder.add_conditional_edges(
    "propose",
    should_execute,
    {
        "human_gate": "human_gate",
        END: END,
    }
)
builder.add_edge("human_gate", "execute")
builder.add_edge("execute", END)

# Checkpointer is mandatory for interrupt & resume
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
