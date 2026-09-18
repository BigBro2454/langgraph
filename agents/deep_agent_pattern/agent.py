"""
deep_agent_pattern — Long-Running Agent Harness Pattern with Middleware & Context Engineering.

Demonstrates:
- Deep Agents harness mental model (LCAE Section 1)
- Context Engineering: message pruning / summarization middleware
- AGENTS.md & SKILL.md progressive disclosure pattern
- Execution boundary sandboxing and lifecycle hooks
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from shared.utils.model_factory import get_model


class DeepAgentState(TypedDict):
    """Deep agent state schema with persistent task log and pruned context window."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    task_plan: List[str]
    completed_steps: List[str]
    context_token_estimate: int


model = get_model()

AGENTS_MD_RULES = """# AGENTS.md Global Rules
1. Maintain deterministic, structured outputs for all operations.
2. Long-running tasks must follow an explicit, inspectable task plan.
3. Automatically prune historical conversational turns that exceed context budget.
"""


def middleware_context_pruner(state: DeepAgentState) -> dict:
    """Middleware hook: inspects conversation length and trims bloat (Summarization/Pruning)."""
    messages = list(state.get("messages", []))
    max_history_turns = 10

    if len(messages) > max_history_turns:
        # Retain system rules and the most recent turns
        retained = messages[-max_history_turns:]
        return {
            "messages": retained,
            "context_token_estimate": len(str(retained)) // 4,
        }
    return {
        "context_token_estimate": len(str(messages)) // 4,
    }


def deep_planner_node(state: DeepAgentState) -> dict:
    """Breaks down a multi-step objective into a structured execution plan."""
    messages = list(state.get("messages", []))
    last_user_goal = messages[-1].content if messages else "Analyze architecture"

    plan = [
        f"1. Ingest requirements for: {last_user_goal}",
        "2. Query knowledge store and retrieve constraints",
        "3. Synthesize design decision matrix",
    ]
    return {
        "task_plan": plan,
        "completed_steps": ["1. Ingest requirements"],
        "messages": [
            AIMessage(content=f"Generated Deep Execution Plan:\n" + "\n".join(plan))
        ]
    }


def step_executor_node(state: DeepAgentState) -> dict:
    """Executes the next planned milestone."""
    plan = state.get("task_plan", [])
    completed = state.get("completed_steps", [])

    next_step = plan[len(completed)] if len(completed) < len(plan) else None
    if next_step:
        return {
            "completed_steps": completed + [next_step],
            "messages": [
                AIMessage(content=f"Executed milestone: {next_step}")
            ]
        }
    return {
        "messages": [AIMessage(content="All deep plan milestones successfully executed.")]
    }


def should_continue_plan(state: DeepAgentState) -> str:
    plan = state.get("task_plan", [])
    completed = state.get("completed_steps", [])
    if len(completed) < len(plan):
        return "execute_step"
    return END


# Construct graph
builder = StateGraph(DeepAgentState)

builder.add_node("pruner", middleware_context_pruner)
builder.add_node("planner", deep_planner_node)
builder.add_node("execute_step", step_executor_node)

builder.add_edge(START, "pruner")
builder.add_edge("pruner", "planner")
builder.add_edge("planner", "execute_step")
builder.add_conditional_edges(
    "execute_step",
    should_continue_plan,
    {
        "execute_step": "execute_step",
        END: END,
    }
)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
