"""
Monte Carlo Tree Search (MCTS) Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 2.5
"Monte Carlo Tree Search (MCTS) uses a four-phase loop—Selection, Expansion,
Simulation, and Backpropagation—to explore combinatorial decision spaces.
Nodes are selected using the Upper Confidence Bound for Trees (UCB1) formula,
balancing exploitation of known high-reward paths with exploration of less-visited
branches."

Pattern Card:
- Type: Single-Agent, Search-Based Optimization
- Best For: High-stakes code generation, competitive problem solving, complex multi-path reasoning
- Complexity: Very High
- Performance Lift: 42-49% on SWE-Bench Verified; 2x solve rate on competitive reasoning
- Key Risk: High token consumption; requires disciplined simulation depth and iteration bounds
"""

import math
import json
from typing import Dict, List, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class MCTSNode(TypedDict):
    id: str
    parent_id: Optional[str]
    action: str
    visits: int
    total_reward: float
    children_ids: List[str]


class MCTSState(TypedDict):
    """Graph state tracking the search tree and current phase."""
    problem: str
    tree: Dict[str, MCTSNode]
    selected_node_id: Optional[str]
    expanded_node_id: Optional[str]
    simulation_reward: float
    current_iteration: int
    max_iterations: int
    best_solution: Optional[str]


model = get_model()

C_PARAM = 1.414  # UCB1 exploration constant

EXPANSION_PROMPT = """You are an MCTS Candidate Expansion Engine.
Given a problem and the current reasoning path, generate 2 diverse, concrete NEXT STEPS or tactical actions.

Output strictly JSON:
{
    "actions": [
        "First specific tactical action/thought",
        "Second specific tactical action/thought"
    ]
}
"""

SIMULATION_PROMPT = """You are an MCTS Rollout and Quality Evaluator.
Given the problem and a reasoning step, perform a rapid mental simulation of whether this path
will lead to an optimal, correct, and robust resolution.

Rate the terminal reward as a float between 0.0 (total failure) and 1.0 (flawless success).
Output strictly JSON:
{
    "reward": 0.85,
    "rationale": "Directly tackles root cause with minimal overhead"
}
"""


def ucb1(node: MCTSNode, parent_visits: int) -> float:
    """Computes Upper Confidence Bound for Trees (UCB1)."""
    if node["visits"] == 0:
        return float("inf")  # Ensure unvisited nodes are explored first
    exploitation = node["total_reward"] / node["visits"]
    exploration = C_PARAM * math.sqrt(math.log(parent_visits) / node["visits"])
    return exploitation + exploration


def selection_node(state: MCTSState) -> dict:
    """Traverses tree from root to find an expandable or unvisited leaf using UCB1."""
    tree = dict(state.get("tree", {}))
    problem = state["problem"]
    iteration = state.get("current_iteration", 0) + 1

    # Initialize root if empty
    if "root" not in tree:
        tree["root"] = {
            "id": "root",
            "parent_id": None,
            "action": "Initial Problem Statement: " + problem,
            "visits": 1,
            "total_reward": 0.5,
            "children_ids": [],
        }

    current_id = "root"
    while tree[current_id]["children_ids"]:
        parent = tree[current_id]
        children = [tree[cid] for cid in parent["children_ids"]]
        # Pick child with highest UCB1
        best_child = max(children, key=lambda c: ucb1(c, parent["visits"]))
        current_id = best_child["id"]
        # If this child hasn't been visited yet, select it for simulation
        if best_child["visits"] == 0:
            break

    return {
        "tree": tree,
        "selected_node_id": current_id,
        "current_iteration": iteration,
    }


def expansion_node(state: MCTSState) -> dict:
    """Generates child nodes for the selected node."""
    tree = dict(state["tree"])
    selected_id = state["selected_node_id"]
    selected_node = tree[selected_id]
    problem = state["problem"]

    # If selected node already has children or is deep, skip expansion
    if selected_node["children_ids"]:
        return {"expanded_node_id": selected_id}

    # Trace back path to root
    path_actions = []
    curr = selected_node
    while curr:
        path_actions.append(curr["action"])
        curr = tree.get(curr["parent_id"]) if curr["parent_id"] else None
    path_actions.reverse()

    user_msg = (
        f"Problem: {problem}\n\n"
        f"Path so far:\n" + "\n".join([f"- {a}" for a in path_actions]) +
        "\n\nPropose 2 next candidate actions in JSON."
    )

    resp = model.invoke([
        SystemMessage(content=EXPANSION_PROMPT),
        HumanMessage(content=user_msg),
    ])

    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    actions = []
    try:
        data = json.loads(text)
        actions = data.get("actions", [])
    except Exception:
        actions = ["Validate assumptions", "Synthesize solution candidate"]

    new_children_ids = []
    for idx, action_text in enumerate(actions[:2]):
        child_id = f"{selected_id}_c{len(selected_node['children_ids']) + idx + 1}"
        tree[child_id] = {
            "id": child_id,
            "parent_id": selected_id,
            "action": str(action_text),
            "visits": 0,
            "total_reward": 0.0,
            "children_ids": [],
        }
        new_children_ids.append(child_id)

    tree[selected_id]["children_ids"].extend(new_children_ids)
    # Pick the first child as the newly expanded node for simulation
    expanded_id = new_children_ids[0] if new_children_ids else selected_id

    return {
        "tree": tree,
        "expanded_node_id": expanded_id,
    }


def simulation_node(state: MCTSState) -> dict:
    """Estimates terminal reward of the expanded node via rollout evaluation."""
    tree = state["tree"]
    expanded_id = state.get("expanded_node_id", "root")
    node = tree[expanded_id]
    problem = state["problem"]

    eval_msg = (
        f"Problem: {problem}\n"
        f"Candidate Step: {node['action']}\n"
        "Evaluate terminal success probability (reward 0.0 to 1.0) in JSON."
    )

    resp = model.invoke([
        SystemMessage(content=SIMULATION_PROMPT),
        HumanMessage(content=eval_msg),
    ])

    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    reward = 0.5
    try:
        data = json.loads(text)
        reward = float(data.get("reward", 0.5))
    except Exception:
        reward = 0.7

    return {"simulation_reward": max(0.0, min(1.0, reward))}


def backpropagation_node(state: MCTSState) -> dict:
    """Propagates simulation reward and increments visit counts up to root."""
    tree = dict(state["tree"])
    current_id = state.get("expanded_node_id", "root")
    reward = state.get("simulation_reward", 0.5)

    while current_id:
        node = tree[current_id]
        node["visits"] += 1
        node["total_reward"] += reward
        current_id = node.get("parent_id")

    return {"tree": tree}


def extract_best_path_node(state: MCTSState) -> dict:
    """Extracts the highest-confidence trajectory and generates the final response."""
    tree = state["tree"]
    problem = state["problem"]

    # Greedily follow children with highest visit count
    trajectory = []
    curr_id = "root"
    while tree[curr_id]["children_ids"]:
        children = [tree[cid] for cid in tree[curr_id]["children_ids"]]
        best_child = max(children, key=lambda c: c["visits"])
        trajectory.append(best_child["action"])
        curr_id = best_child["id"]

    prompt = (
        f"Problem: {problem}\n\n"
        f"Optimal MCTS Trajectory (Discovered through systematic search):\n"
        + "\n".join([f"- Step {i+1}: {step}" for i, step in enumerate(trajectory)])
        + "\n\nSynthesize the definitive solution and final answer for the user."
    )

    resp = model.invoke([HumanMessage(content=prompt)])
    return {"best_solution": str(resp.content)}


def should_continue_mcts(state: MCTSState) -> str:
    """Checks if iteration budget has been exhausted."""
    if state.get("current_iteration", 0) >= state.get("max_iterations", 3):
        return "extract"
    return "select"


builder = StateGraph(MCTSState)

builder.add_node("selection", selection_node)
builder.add_node("expansion", expansion_node)
builder.add_node("simulation", simulation_node)
builder.add_node("backpropagation", backpropagation_node)
builder.add_node("extract_best_path", extract_best_path_node)

builder.add_edge(START, "selection")
builder.add_edge("selection", "expansion")
builder.add_edge("expansion", "simulation")
builder.add_edge("simulation", "backpropagation")
builder.add_conditional_edges(
    "backpropagation",
    should_continue_mcts,
    {
        "select": "selection",
        "extract": "extract_best_path",
    },
)
builder.add_edge("extract_best_path", END)

graph = builder.compile()
