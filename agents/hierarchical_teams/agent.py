"""
Hierarchical Teams Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 5.2
"Hierarchical Teams extend Supervisor-Crew to multiple levels—a top-level supervisor
delegates to team leads, who in turn manage their own crews of workers. This creates
a tree-structured organization that mirrors how human teams scale. Natural domain
isolation—each team manages its own context."

Pattern Card:
- Type: Multi-Agent, Multi-Level Hierarchy (Tree-Structured Organization)
- Best For: Complex cross-functional projects (10+ agents), research + engineering pipelines
- Complexity: High
- Performance Lift: Scales linearly across complex enterprise domains without context blowup
- Key Risk: Communication overhead between tiers; potential information loss across abstractions
"""

from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model
from shared.tools.web_search import search_knowledge_base


class HierarchicalState(TypedDict):
    """Global state across hierarchical team levels."""
    query: str
    research_brief: Optional[str]
    engineering_spec: Optional[str]
    executive_deliverable: Optional[str]


model = get_model()

# ==========================================
# Level 1: Sub-Team A — Research Team
# ==========================================

def research_lead_node(state: HierarchicalState) -> dict:
    """Research Lead: orchestrates market research and fact extraction."""
    query = state["query"]
    # Worker 1: Fact Retrieval
    raw_data = search_knowledge_base.invoke({"query": query})

    prompt = (
        f"You are the Research Team Lead.\n"
        f"Objective: {query}\n"
        f"Raw Domain Data: {raw_data}\n\n"
        "Synthesize a structured research brief highlighting 3 critical domain insights, "
        "current state-of-the-art benchmarks, and industry precedents."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"research_brief": str(resp.content)}


# ==========================================
# Level 1: Sub-Team B — Engineering Team
# ==========================================

def engineering_lead_node(state: HierarchicalState) -> dict:
    """Engineering Lead: designs technical architecture based on research brief."""
    query = state["query"]
    research = state.get("research_brief", "No research provided.")

    prompt = (
        f"You are the Principal Systems Engineering Lead.\n"
        f"Goal: {query}\n"
        f"Input from Research Team:\n{research}\n\n"
        "Specify the technical system architecture:\n"
        "1. Core State Schema & Storage\n"
        "2. Node & Edge Graph Topology\n"
        "3. Failure Recovery & Guardrail Mechanisms"
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"engineering_spec": str(resp.content)}


# ==========================================
# Level 0: Top-Level Meta-Orchestrator
# ==========================================

def executive_director_node(state: HierarchicalState) -> dict:
    """Executive Director: synthesizes deliverables across all hierarchical teams."""
    query = state["query"]
    research = state.get("research_brief", "N/A")
    engineering = state.get("engineering_spec", "N/A")

    prompt = (
        f"You are the Executive Director coordinating cross-functional AI teams.\n"
        f"Original User Mission: {query}\n\n"
        f"--- Research Team Deliverable ---\n{research}\n\n"
        f"--- Engineering Team Deliverable ---\n{engineering}\n\n"
        "Synthesize a cohesive, high-impact Executive Product & Engineering Roadmap "
        "ready for board and leadership review."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"executive_deliverable": str(resp.content)}


builder = StateGraph(HierarchicalState)

builder.add_node("research_team", research_lead_node)
builder.add_node("engineering_team", engineering_lead_node)
builder.add_node("executive_director", executive_director_node)

builder.add_edge(START, "research_team")
builder.add_edge("research_team", "engineering_team")
builder.add_edge("engineering_team", "executive_director")
builder.add_edge("executive_director", END)

graph = builder.compile()
