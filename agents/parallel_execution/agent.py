"""
Parallel Execution Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 5.6
"Multiple agents work on different aspects of a task simultaneously, with results
aggregated at the end. This pattern trades cost for latency and reliability.
Dramatically reduces latency for parallelizable tasks and provides natural fault
tolerance through ensemble aggregation."

Pattern Card:
- Type: Multi-Agent, Fan-Out / Fan-In Concurrent Architecture
- Best For: Multi-dimensional audits, latency-critical multi-perspective analysis, ensemble voting
- Complexity: Medium
- Performance Lift: Near-constant latency O(max(worker_latencies)) instead of O(sum); ensemble accuracy
- Key Risk: Cost multiplies linearly by parallel agent count; requires robust aggregator
"""

from typing import TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class ParallelState(TypedDict):
    """Graph state gathering results from concurrent specialist nodes."""
    proposal: str
    security_audit: Optional[str]
    performance_audit: Optional[str]
    cost_audit: Optional[str]
    synthesized_scorecard: Optional[str]


model = get_model()


def security_audit_node(state: ParallelState) -> dict:
    """Specialist 1: Parallel security & vulnerability assessment."""
    proposal = state["proposal"]
    prompt = (
        f"You are the Principal Security Architect.\n"
        f"Analyze this architectural proposal for threat vectors, data leakage, and authentication risks:\n\n"
        f"{proposal}\n\n"
        "Provide a concise security audit with a risk rating (LOW/MEDIUM/HIGH)."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"security_audit": str(resp.content)}


def performance_audit_node(state: ParallelState) -> dict:
    """Specialist 2: Parallel performance & latency bottleneck analysis."""
    proposal = state["proposal"]
    prompt = (
        f"You are the Principal Performance & Scale Engineer.\n"
        f"Analyze this architectural proposal for concurrency bottlenecks, throughput ceilings, and latency hot spots:\n\n"
        f"{proposal}\n\n"
        "Provide a concise performance audit with expected latency characteristics."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"performance_audit": str(resp.content)}


def cost_audit_node(state: ParallelState) -> dict:
    """Specialist 3: Parallel cloud infrastructure & LLM token cost analysis."""
    proposal = state["proposal"]
    prompt = (
        f"You are the FinOps & Cloud Economics Specialist.\n"
        f"Analyze this architectural proposal for LLM token usage, infrastructure overhead, and cost efficiency:\n\n"
        f"{proposal}\n\n"
        "Provide a concise FinOps audit with estimated relative cost tier."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"cost_audit": str(resp.content)}


def fan_in_synthesizer_node(state: ParallelState) -> dict:
    """Fan-In Aggregator: synthesizes parallel audits into a unified scorecard."""
    proposal = state["proposal"]
    sec = state.get("security_audit", "Pending")
    perf = state.get("performance_audit", "Pending")
    cost = state.get("cost_audit", "Pending")

    prompt = (
        f"You are the Chief Technology Officer (CTO).\n"
        f"Original Proposal:\n{proposal}\n\n"
        f"--- Parallel Security Audit ---\n{sec}\n\n"
        f"--- Parallel Performance Audit ---\n{perf}\n\n"
        f"--- Parallel FinOps Cost Audit ---\n{cost}\n\n"
        "Synthesize these parallel audits into an Executive Multi-Dimensional Scorecard: "
        "1. Executive Verdict (Approve / Reject / Amend)\n"
        "2. Scorecard Table (Security, Performance, Cost)\n"
        "3. Top Recommended Mitigations"
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"synthesized_scorecard": str(resp.content)}


builder = StateGraph(ParallelState)

# Nodes
builder.add_node("security_worker", security_audit_node)
builder.add_node("performance_worker", performance_audit_node)
builder.add_node("cost_worker", cost_audit_node)
builder.add_node("synthesizer", fan_in_synthesizer_node)

# Fan-Out: START triggers all 3 workers in parallel
builder.add_edge(START, "security_worker")
builder.add_edge(START, "performance_worker")
builder.add_edge(START, "cost_worker")

# Fan-In: All 3 workers converge into synthesizer
builder.add_edge("security_worker", "synthesizer")
builder.add_edge("performance_worker", "synthesizer")
builder.add_edge("cost_worker", "synthesizer")

builder.add_edge("synthesizer", END)

graph = builder.compile()
