"""
Tree-of-Thought (ToT) Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 2.4
"Tree-of-Thought models reasoning as search through a tree where each node represents
a coherent thought (an intermediate reasoning step) and each edge represents a transition.
At each step, the agent generates K candidate thoughts, evaluates their promise using an
evaluation function, and uses a search algorithm (BFS/DFS) to explore the most promising paths."

Pattern Card:
- Type: Single-Agent, Search-Based Reasoning (Branching & Pruning)
- Best For: Creative exploration, constraint satisfaction, strategic trade-off analysis
- Complexity: High
- Performance Lift: 74% vs 4% on Game of 24; 65-75% preference on multi-constraint tasks
- Key Risk: Token cost scales with K * depth; requires well-calibrated evaluation function
"""

import json
from typing import List, Dict, Any, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class ThoughtPath(TypedDict):
    path_id: str
    steps: List[str]
    score: float
    reasoning: str


class ToTState(TypedDict):
    """State schema for Tree-of-Thought search."""
    problem: str
    depth: int
    max_depth: int
    active_paths: List[ThoughtPath]
    candidates: List[ThoughtPath]
    best_path: Optional[ThoughtPath]
    final_output: Optional[str]


model = get_model()

BRANCHING_FACTOR = 3  # K = 3 candidate thoughts per active path
BEAM_WIDTH = 2        # B = 2 top paths retained per level


PROPOSER_PROMPT = """You are a Divergent Thought Generation Engine.
Given a complex challenge and a partial path of intermediate thoughts, generate 3 distinct,
logically sound candidate NEXT STEPS or angles to advance toward solving the problem.

Output strictly JSON:
{
    "candidates": [
        "First distinct candidate step/angle",
        "Second distinct candidate step/angle",
        "Third distinct candidate step/angle"
    ]
}
"""

EVALUATOR_PROMPT = """You are an Objective Evaluator of Reasoning Paths.
Evaluate each proposed thought path for:
1. Feasibility and logical validity.
2. Direct contribution toward resolving the core problem.
3. Lack of contradictions or hallucinations.

Assign a score from 1.0 (dead end) to 10.0 (optimal) for each candidate.
Output strictly JSON:
{
    "evaluations": [
        {"index": 0, "score": 8.5, "rationale": "High impact, practical"},
        {"index": 1, "score": 4.0, "rationale": "High risk of failure"}
    ]
}
"""


def branch_node(state: ToTState) -> dict:
    """Proposes K candidate next thoughts for each active path."""
    problem = state["problem"]
    active = state.get("active_paths", [])
    depth = state.get("depth", 0)

    # Initial root expansion if no active paths yet
    if not active:
        active = [{"path_id": "root", "steps": [], "score": 0.0, "reasoning": "Initial state"}]

    candidates: List[ThoughtPath] = []
    for p_idx, path in enumerate(active):
        context = "\n".join([f"- Step {i+1}: {step}" for i, step in enumerate(path["steps"])])
        user_msg = (
            f"Problem: {problem}\n\n"
            f"Current Thought Path ({path['path_id']}):\n{context or '(None - beginning analysis)'}\n"
            f"Target Depth: {depth + 1}\n\n"
            "Generate 3 diverse candidate next thoughts."
        )

        resp = model.invoke([
            SystemMessage(content=PROPOSER_PROMPT),
            HumanMessage(content=user_msg),
        ])

        text = str(resp.content)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            cands = data.get("candidates", [])
            for c_idx, cand in enumerate(cands[:BRANCHING_FACTOR]):
                new_steps = list(path["steps"]) + [cand]
                candidates.append({
                    "path_id": f"{path['path_id']}.{c_idx+1}",
                    "steps": new_steps,
                    "score": 0.0,
                    "reasoning": "",
                })
        except Exception:
            candidates.append({
                "path_id": f"{path['path_id']}.1",
                "steps": list(path["steps"]) + [f"Deepen analysis on: {problem}"],
                "score": 5.0,
                "reasoning": "Fallback branch",
            })

    return {"candidates": candidates}


def evaluate_node(state: ToTState) -> dict:
    """Scores candidate thought paths using heuristic/LLM rubric."""
    problem = state["problem"]
    candidates = state.get("candidates", [])

    if not candidates:
        return {"candidates": []}

    formatted_candidates = []
    for idx, c in enumerate(candidates):
        steps_str = " -> ".join(c["steps"])
        formatted_candidates.append(f"Candidate {idx} [ID: {c['path_id']}]:\nSteps: {steps_str}")

    prompt = (
        f"Problem: {problem}\n\n"
        f"Candidate Reasoning Paths:\n\n" + "\n\n".join(formatted_candidates) +
        "\n\nEvaluate each candidate and score from 1.0 to 10.0 in JSON."
    )

    resp = model.invoke([
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=prompt),
    ])

    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    evaluated_candidates = list(candidates)
    try:
        data = json.loads(text)
        evals = data.get("evaluations", [])
        for ev in evals:
            idx = ev.get("index")
            if idx is not None and 0 <= idx < len(evaluated_candidates):
                evaluated_candidates[idx]["score"] = float(ev.get("score", 5.0))
                evaluated_candidates[idx]["reasoning"] = ev.get("rationale", "")
    except Exception:
        for idx, c in enumerate(evaluated_candidates):
            c["score"] = 6.0 + (idx * 0.5)

    return {"candidates": evaluated_candidates}


def prune_node(state: ToTState) -> dict:
    """Selects top B candidate paths (beam search) and increments depth."""
    candidates = state.get("candidates", [])
    current_depth = state.get("depth", 0) + 1

    # Sort descending by score
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    surviving_paths = sorted_candidates[:BEAM_WIDTH]

    best_candidate = surviving_paths[0] if surviving_paths else None

    return {
        "active_paths": surviving_paths,
        "best_path": best_candidate,
        "depth": current_depth,
        "candidates": [],
    }


def synthesize_node(state: ToTState) -> dict:
    """Synthesizes the winning reasoning path into a final cohesive output."""
    problem = state["problem"]
    best_path = state.get("best_path")

    if not best_path:
        steps_summary = "Standard analytical breakdown."
    else:
        steps_summary = "\n".join([f"{i+1}. {step}" for i, step in enumerate(best_path["steps"])])

    prompt = (
        f"Original Problem: {problem}\n\n"
        f"Highest-Scored Reasoning Path (Score: {best_path['score'] if best_path else 'N/A'}/10):\n"
        f"{steps_summary}\n\n"
        "Synthesize this optimal path into a clear, detailed, and polished executive solution."
    )

    resp = model.invoke([HumanMessage(content=prompt)])
    return {"final_output": str(resp.content)}


def should_continue_tot(state: ToTState) -> str:
    """Checks if search depth has reached target max depth."""
    if state.get("depth", 0) >= state.get("max_depth", 2):
        return "synthesize"
    return "branch"


builder = StateGraph(ToTState)

builder.add_node("branch", branch_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("prune", prune_node)
builder.add_node("synthesize", synthesize_node)

builder.add_edge(START, "branch")
builder.add_edge("branch", "evaluate")
builder.add_edge("evaluate", "prune")
builder.add_conditional_edges(
    "prune",
    should_continue_tot,
    {
        "branch": "branch",
        "synthesize": "synthesize",
    },
)
builder.add_edge("synthesize", END)

graph = builder.compile()
