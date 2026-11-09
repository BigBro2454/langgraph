"""
Plan-and-Execute Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 2.2
"Plan-and-Execute separates planning from execution into two distinct phases.
A planner LLM first decomposes the task into an ordered list of subtasks,
then an executor agent handles each subtask sequentially. After each subtask
completes, the planner can optionally revise the remaining plan based on new information."

Pattern Card:
- Type: Single-Agent, Two-Tier Orchestration (Strategic Planning + Tactical Execution)
- Best For: Complex multi-step tasks (5-20 steps), research workflows, code refactoring
- Complexity: Medium
- Performance Lift: +25-40% improvement over basic ReAct on complex tasks
- Key Risk: Over-planning upfront; poor initial plan quality; replanning loop budget
"""

import json
from typing import List, Tuple, TypedDict, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model
from shared.tools.calculator import add, multiply
from shared.tools.web_search import search_knowledge_base


class Plan(BaseModel):
    """Structured plan composed of ordered subtasks."""
    steps: List[str] = Field(
        description="Ordered list of concise, actionable steps to achieve the goal."
    )


class ActResponse(BaseModel):
    """Replanner response: either revise the remaining steps or declare the objective finished."""
    response: Optional[str] = Field(
        default=None,
        description="Final comprehensive answer if the objective is fully achieved."
    )
    remaining_steps: Optional[List[str]] = Field(
        default=None,
        description="Updated list of remaining subtasks to execute. Empty if completed."
    )


class PlanExecuteState(TypedDict):
    """Graph state tracking the strategic plan, execution trail, and final response."""
    input: str
    plan: List[str]
    past_steps: List[Tuple[str, str]]
    response: Optional[str]
    step_count: int


model = get_model()

PLANNER_PROMPT = """You are a Strategic Planning Agent.
Given a complex objective, decompose it into a clear, minimal, sequential plan of 2 to 4 actionable steps.
Output strictly JSON matching: {"steps": ["step 1", "step 2", ...]}
Do not include steps that have already been accomplished.
"""

REPLANNER_PROMPT = """You are an Adaptive Replanner Agent.
Analyze the original user objective, the completed steps with their results, and the pending plan.

Decide:
1. If the goal is satisfied, provide the final answer in "response" and set "remaining_steps": [].
2. If more steps are needed, provide the updated list in "remaining_steps" and "response": null.

Respond strictly in valid JSON format:
{"response": "...", "remaining_steps": [...]}
"""

MAX_EXECUTION_STEPS = 6


def plan_node(state: PlanExecuteState) -> dict:
    """Generates the initial strategic plan from the user prompt."""
    user_input = state["input"]
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Objective: {user_input}"),
    ]
    resp = model.invoke(messages)
    content = resp.content

    # Clean JSON
    text = str(content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        steps = data.get("steps", [])
        if not steps or not isinstance(steps, list):
            steps = [f"Execute task: {user_input}"]
    except Exception:
        # Fallback plan
        steps = [
            f"Research and collect information on: {user_input}",
            f"Synthesize findings and provide final answer for: {user_input}",
        ]

    return {"plan": steps, "past_steps": [], "step_count": 0}


def execute_step_node(state: PlanExecuteState) -> dict:
    """Executes the next available step in the active plan."""
    plan = state["plan"]
    past_steps = list(state.get("past_steps", []))
    step_count = state.get("step_count", 0) + 1

    if not plan:
        return {"step_count": step_count}

    current_step = plan[0]

    # Execute step with LLM context and tools
    context_str = "\n".join([f"Step: {s}\nResult: {r}" for s, r in past_steps])
    prompt = (
        f"Goal: {state['input']}\n"
        f"Past Step Context:\n{context_str or 'None'}\n\n"
        f"Current Subtask to complete: {current_step}\n"
        f"Execute this subtask accurately and provide the finding/result."
    )

    resp = model.invoke([HumanMessage(content=prompt)])
    result_text = str(resp.content)

    past_steps.append((current_step, result_text))
    # Remove completed step from plan
    remaining_plan = plan[1:]

    return {
        "plan": remaining_plan,
        "past_steps": past_steps,
        "step_count": step_count,
    }


def replan_node(state: PlanExecuteState) -> dict:
    """Assesses progress and either updates remaining plan or drafts final output."""
    plan = state.get("plan", [])
    past_steps = state.get("past_steps", [])
    step_count = state.get("step_count", 0)

    # Circuit breaker
    if step_count >= MAX_EXECUTION_STEPS or not plan:
        # Formulate final synthesis
        summary_prompt = (
            f"Original Objective: {state['input']}\n"
            f"Execution Steps & Results:\n"
            + "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in past_steps])
            + "\n\nSynthesize the final, comprehensive response answering the objective."
        )
        resp = model.invoke([HumanMessage(content=summary_prompt)])
        return {"response": str(resp.content), "plan": []}

    steps_context = "\n".join([f"- Step: {s}\n  Result: {r}" for s, r in past_steps])
    prompt = (
        f"Objective: {state['input']}\n\n"
        f"Completed Steps:\n{steps_context}\n\n"
        f"Pending Steps: {plan}\n\n"
        f"Evaluate if more work is needed. If complete, write final response in JSON."
    )

    resp = model.invoke([
        SystemMessage(content=REPLANNER_PROMPT),
        HumanMessage(content=prompt),
    ])

    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        final_resp = data.get("response")
        updated_plan = data.get("remaining_steps")
        if final_resp:
            return {"response": final_resp, "plan": []}
        elif updated_plan is not None and isinstance(updated_plan, list):
            return {"plan": updated_plan}
    except Exception:
        pass

    # If still steps remaining in current plan, keep going
    return {"plan": plan}


def should_replan_continue(state: PlanExecuteState) -> str:
    """Routes back to step execution if subtasks remain, or END if finished."""
    if state.get("response") is not None:
        return "finish"
    if not state.get("plan"):
        return "finish"
    return "execute"


builder = StateGraph(PlanExecuteState)

builder.add_node("planner", plan_node)
builder.add_node("executor", execute_step_node)
builder.add_node("replanner", replan_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", "replanner")
builder.add_conditional_edges(
    "replanner",
    should_replan_continue,
    {
        "execute": "executor",
        "finish": END,
    },
)

graph = builder.compile()
