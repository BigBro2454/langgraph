"""
Reflexion Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 2.3
"Reflexion equips an agent with dynamic memory and self-reflection capabilities.
When an action fails or produces suboptimal results, the agent generates a verbal
self-reflection analyzing why the failure occurred and what to do differently.
This reflection is stored in episodic memory and conditioned upon in subsequent trials."

Pattern Card:
- Type: Single-Agent, Reinforcement via Verbal Self-Reflection
- Best For: Code generation & debugging, verifiable math, tasks with structured rubrics
- Complexity: Medium-High
- Performance Lift: +36% relative on HumanEval (67% -> 91%); 30-38% on SWE-Bench
- Key Risk: Needs reliable evaluation function; cost scales with trials; reflection loops
"""

import json
from typing import List, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class ReflexionState(TypedDict):
    """State schema for Reflexion Agent."""
    task: str
    current_trial: str
    evaluation_passed: bool
    evaluation_feedback: str
    reflections: List[str]
    trial_count: int


model = get_model()
MAX_TRIALS = 3

ACTOR_PROMPT = """You are an Expert Problem-Solving Actor Agent.
Your goal is to solve the given programming or algorithmic task accurately.
If you are provided with prior failed trials and self-reflections, CAREFULLY review the mistakes
noted in the reflections and ensure your new solution directly corrects them.

Always output your clean, complete solution with code and explanatory docstrings.
"""

EVALUATOR_PROMPT = """You are a Strict Code & Quality Evaluator Agent.
Review the proposed solution against the original task requirements.
Check:
1. Algorithmic correctness and logic.
2. Edge cases (empty inputs, negative values, boundary conditions).
3. Syntax validity and complete runnable implementation.

Output strictly JSON:
{
    "passed": true/false,
    "score": 1-10,
    "critique": "Detailed explanation of what failed or why it succeeded."
}
"""

REFLECTOR_PROMPT = """You are an Introspective Self-Reflection Agent.
The previous trial failed the evaluation criteria.
Analyze:
1. What was the exact root cause of the failure?
2. What specific logical flaw or blind spot occurred?
3. What concrete action should the actor take in the next trial to fix it?

Provide a concise, highly specific 2-3 sentence reflection lesson.
"""


def actor_node(state: ReflexionState) -> dict:
    """Generates a solution conditioned on past failed trials and reflection memory."""
    task = state["task"]
    reflections = state.get("reflections", [])
    trial = state.get("trial_count", 0) + 1

    prompt_parts = [f"Task Objective:\n{task}\n"]
    if reflections:
        prompt_parts.append("\nEpisodic Reflection Memory (Lessons Learned from Previous Failures):")
        for idx, ref in enumerate(reflections, 1):
            prompt_parts.append(f"{idx}. {ref}")
        prompt_parts.append(
            "\nPay special attention to these past errors and construct a robust, corrected solution."
        )

    messages = [
        SystemMessage(content=ACTOR_PROMPT),
        HumanMessage(content="\n".join(prompt_parts)),
    ]
    resp = model.invoke(messages)
    return {
        "current_trial": str(resp.content),
        "trial_count": trial,
    }


def evaluator_node(state: ReflexionState) -> dict:
    """Evaluates the candidate solution against objective correctness criteria."""
    task = state["task"]
    trial_content = state["current_trial"]

    eval_input = (
        f"Original Task:\n{task}\n\n"
        f"Proposed Candidate Solution (Trial #{state.get('trial_count', 1)}):\n{trial_content}\n"
    )

    resp = model.invoke([
        SystemMessage(content=EVALUATOR_PROMPT),
        HumanMessage(content=eval_input),
    ])

    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    passed = False
    feedback = "Evaluation parsing failed."
    try:
        data = json.loads(text)
        passed = bool(data.get("passed", False))
        feedback = data.get("critique", "No critique provided.")
    except Exception:
        # Fallback heuristic: check if score was reported as passing
        if '"passed": true' in text.lower() or "passed: true" in text.lower():
            passed = True
            feedback = "Pass criteria met."
        else:
            feedback = text[:200]

    return {
        "evaluation_passed": passed,
        "evaluation_feedback": feedback,
    }


def reflector_node(state: ReflexionState) -> dict:
    """Reflects on the failure and appends lessons learned to reflection memory."""
    task = state["task"]
    trial_content = state["current_trial"]
    feedback = state["evaluation_feedback"]
    reflections = list(state.get("reflections", []))

    reflection_input = (
        f"Task:\n{task}\n\n"
        f"Failed Solution:\n{trial_content}\n\n"
        f"Evaluator Critique:\n{feedback}\n"
    )

    resp = model.invoke([
        SystemMessage(content=REFLECTOR_PROMPT),
        HumanMessage(content=reflection_input),
    ])

    reflection_note = str(resp.content).strip()
    reflections.append(f"[Trial {state['trial_count']} Insight]: {reflection_note}")

    return {"reflections": reflections}


def should_retry(state: ReflexionState) -> str:
    """Routes to END if passed or max trials reached, else reflects for next trial."""
    if state.get("evaluation_passed", False):
        return "passed"
    if state.get("trial_count", 0) >= MAX_TRIALS:
        return "max_trials_exhausted"
    return "reflect"


builder = StateGraph(ReflexionState)

builder.add_node("actor", actor_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("reflector", reflector_node)

builder.add_edge(START, "actor")
builder.add_edge("actor", "evaluator")
builder.add_conditional_edges(
    "evaluator",
    should_retry,
    {
        "passed": END,
        "max_trials_exhausted": END,
        "reflect": "reflector",
    },
)
builder.add_edge("reflector", "actor")

graph = builder.compile()
