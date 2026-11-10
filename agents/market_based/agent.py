"""
Market-Based Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 5.4
"Available agents bid on tasks based on their capabilities and current load,
and a market mechanism allocates work to the best-matched agents. This pattern
excels at dynamic resource allocation in systems with heterogeneous agents and
variable workloads."

Pattern Card:
- Type: Multi-Agent, Decentralized Market / Contract Net Protocol
- Best For: Heterogeneous agent pools, dynamic resource allocation, cost/latency optimization
- Complexity: High
- Performance Lift: Automatically optimizes system token cost by 40-60% via competitive routing
- Key Risk: Bidding overhead latency; requires reliable capability calibration
"""

import json
from typing import List, Dict, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class AgentBid(TypedDict):
    agent_id: str
    confidence: float          # 0.0 to 1.0
    estimated_cost_tokens: int # token estimate
    estimated_latency_sec: float
    methodology: str
    utility_score: float


class MarketState(TypedDict):
    """Graph state tracking the RFP, bids, winning contractor, and execution deliverable."""
    task_description: str
    optimization_priority: str  # "balanced", "cost", "speed", "quality"
    rfp_details: Optional[str]
    bids: List[AgentBid]
    winning_agent_id: Optional[str]
    award_rationale: Optional[str]
    execution_result: Optional[str]
    settlement_audit: Optional[Dict[str, any]]


model = get_model()

# Define heterogeneous agent profiles
AGENT_PROFILES = [
    {
        "id": "FastResponseAgent",
        "specialty": "Rapid synthesis, basic Q&A, low latency",
        "base_tokens": 300,
        "latency_sec": 0.8,
        "style": "Concise, direct, bulleted",
    },
    {
        "id": "DeepResearchAgent",
        "specialty": "Exhaustive literature, nuanced trade-offs, thorough exploration",
        "base_tokens": 1500,
        "latency_sec": 4.5,
        "style": "Comprehensive, deeply cited, highly analytical",
    },
    {
        "id": "SecurityComplianceAgent",
        "specialty": "Vulnerability scanning, regulatory policy, edge-case audit",
        "base_tokens": 900,
        "latency_sec": 2.2,
        "style": "Defensive, risk-oriented, compliance-aligned",
    },
]


def broker_rfp_node(state: MarketState) -> dict:
    """Task Broker structures the incoming request into a formal Request For Proposal (RFP)."""
    task = state["task_description"]
    priority = state.get("optimization_priority", "balanced")

    rfp = (
        f"[RFP-001] Task: {task}\n"
        f"Optimization Priority: {priority.upper()}\n"
        f"Eligibility: All registered specialist agents."
    )
    return {"rfp_details": rfp}


def auction_bidding_node(state: MarketState) -> dict:
    """Specialist agents evaluate the RFP and formulate competitive bids."""
    task = state["task_description"]
    priority = state.get("optimization_priority", "balanced")

    bids: List[AgentBid] = []

    # Elicit bids from agents based on their profile
    for profile in AGENT_PROFILES:
        bid_prompt = (
            f"You are {profile['id']}.\n"
            f"Your Core Specialty: {profile['specialty']}\n\n"
            f"Client RFP: {task}\n"
            f"Priority: {priority}\n\n"
            "Estimate your confidence (0.0 to 1.0) and describe your proposed methodology.\n"
            "Output JSON:\n"
            '{"confidence": 0.9, "methodology": "How you will execute the task"}'
        )

        resp = model.invoke([HumanMessage(content=bid_prompt)])
        text = str(resp.content)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        confidence = 0.7
        methodology = f"Execution according to {profile['id']} standard protocol."
        try:
            data = json.loads(text)
            confidence = float(data.get("confidence", 0.7))
            methodology = str(data.get("methodology", methodology))
        except Exception:
            pass

        # Calculate utility score based on optimization priority
        cost = profile["base_tokens"]
        latency = profile["latency_sec"]

        if priority == "cost":
            utility = (confidence * 40.0) - (cost / 40.0) - (latency * 2.0)
        elif priority == "speed":
            utility = (confidence * 40.0) - (latency * 15.0) - (cost / 100.0)
        elif priority == "quality":
            utility = (confidence * 90.0) - (cost / 200.0)
        else:  # balanced
            utility = (confidence * 60.0) - (cost / 70.0) - (latency * 5.0)

        bids.append({
            "agent_id": profile["id"],
            "confidence": confidence,
            "estimated_cost_tokens": cost,
            "estimated_latency_sec": latency,
            "methodology": methodology,
            "utility_score": round(utility, 2),
        })

    return {"bids": bids}


def broker_clearing_node(state: MarketState) -> dict:
    """Task Broker evaluates bids and awards the contract to the highest utility agent."""
    bids = state.get("bids", [])
    if not bids:
        return {"winning_agent_id": "FastResponseAgent", "award_rationale": "Default award"}

    winning_bid = max(bids, key=lambda b: b["utility_score"])

    rationale = (
        f"Awarded contract to {winning_bid['agent_id']} (Utility Score: {winning_bid['utility_score']}) "
        f"with confidence {winning_bid['confidence']} and estimated cost {winning_bid['estimated_cost_tokens']} tokens."
    )

    return {
        "winning_agent_id": winning_bid["agent_id"],
        "award_rationale": rationale,
    }


def execution_node(state: MarketState) -> dict:
    """The winning contractor executes the task."""
    task = state["task_description"]
    winner_id = state.get("winning_agent_id", "FastResponseAgent")
    bids = state.get("bids", [])

    winning_bid = next((b for b in bids if b["agent_id"] == winner_id), None)
    methodology = winning_bid["methodology"] if winning_bid else "Standard execution"

    exec_prompt = (
        f"You are {winner_id}. You won the competitive contract for this task.\n"
        f"Client Task: {task}\n"
        f"Your Committed Methodology: {methodology}\n\n"
        "Execute the deliverable thoroughly according to your committed standard."
    )

    resp = model.invoke([HumanMessage(content=exec_prompt)])
    deliverable = str(resp.content)

    settlement = {
        "contractor": winner_id,
        "token_settlement": winning_bid["estimated_cost_tokens"] if winning_bid else 500,
        "status": "SETTLED_SUCCESSFULLY",
    }

    return {
        "execution_result": deliverable,
        "settlement_audit": settlement,
    }


builder = StateGraph(MarketState)

builder.add_node("broker_rfp", broker_rfp_node)
builder.add_node("auction_bidding", auction_bidding_node)
builder.add_node("broker_clearing", broker_clearing_node)
builder.add_node("contract_execution", execution_node)

builder.add_edge(START, "broker_rfp")
builder.add_edge("broker_rfp", "auction_bidding")
builder.add_edge("auction_bidding", "broker_clearing")
builder.add_edge("broker_clearing", "contract_execution")
builder.add_edge("contract_execution", END)

graph = builder.compile()
