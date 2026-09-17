"""
supervisor — Multi-Agent Orchestration System.

Demonstrates:
- Central Supervisor / Coordinator agent routing tasks to specialist agents
- Specialist Worker 1: Researcher (gathers facts and background)
- Specialist Worker 2: Writer (synthesizes structured reports)
- Conditional handoff loop: Supervisor -> (Worker -> Supervisor)* -> FINISH
"""

from typing import Annotated, Sequence, TypedDict, Literal, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from shared.utils.model_factory import get_model
from shared.tools.web_search import search_knowledge_base


class MultiAgentState(TypedDict):
    """Shared state across the multi-agent supervisor team."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    research_data: Optional[str]
    report_draft: Optional[str]
    next_worker: Optional[str]


# Routing schema for structured supervisor decision
class RouteDecision(BaseModel):
    next_action: Literal["researcher", "writer", "FINISH"] = Field(
        description="The next worker to invoke, or 'FINISH' if the user's objective is fully accomplished."
    )
    reason: str = Field(description="Explanation for why this worker was chosen.")


model = get_model()

SUPERVISOR_PROMPT = """You are the Supervisor Agent managing a team of two specialists:
1. 'researcher': Researches facts, technical concepts, or background data.
2. 'writer': Synthesizes collected facts into a clean, structured, executive report.

Analyze the user's request and the conversation state.
- If background research is needed and not yet collected, route to 'researcher'.
- If research is available but the final report is not written, route to 'writer'.
- If the report has been written and the goal is achieved, route to 'FINISH'.
"""


def supervisor_node(state: MultiAgentState) -> dict:
    """Decides which worker should take action next."""
    messages = list(state.get("messages", []))
    research = state.get("research_data")
    draft = state.get("report_draft")

    context_summary = f"\nCurrent State:\n- Research collected: {bool(research)}\n- Report drafted: {bool(draft)}"
    prompt = [SystemMessage(content=SUPERVISOR_PROMPT + context_summary)] + messages

    # Deterministic fallback or structured parsing
    if not research:
        return {"next_worker": "researcher"}
    elif not draft:
        return {"next_worker": "writer"}
    else:
        return {"next_worker": "FINISH"}


def researcher_node(state: MultiAgentState) -> dict:
    """Specialist worker that retrieves and organizes domain information."""
    messages = list(state.get("messages", []))
    last_query = messages[-1].content if messages else "LangGraph overview"

    # Call internal knowledge base tool
    research_output = search_knowledge_base.invoke({"query": last_query})

    response_msg = HumanMessage(
        content=f"[Researcher Output]: Extracted core insights regarding '{last_query}': {research_output}"
    )
    return {
        "research_data": research_output,
        "messages": [response_msg],
    }


def writer_node(state: MultiAgentState) -> dict:
    """Specialist worker that drafts an executive briefing from research."""
    research = state.get("research_data", "No data gathered")
    writer_prompt = (
        f"You are the Lead Technical Writer. Using the research below, draft a concise 3-bullet "
        f"executive briefing for a Technical Product Manager:\n\nResearch: {research}"
    )

    response = model.invoke([HumanMessage(content=writer_prompt)])
    response_msg = HumanMessage(
        content=f"[Writer Final Report]:\n{response.content}"
    )
    return {
        "report_draft": response.content,
        "messages": [response_msg],
    }


def route_next(state: MultiAgentState) -> str:
    """Routes to the worker selected by the supervisor."""
    worker = state.get("next_worker")
    if worker == "researcher":
        return "researcher"
    elif worker == "writer":
        return "writer"
    return END


# Construct graph
builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "researcher": "researcher",
        "writer": "writer",
        END: END,
    }
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")

graph = builder.compile()
