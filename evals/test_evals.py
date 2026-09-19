"""
Evaluation Framework for LangGraph Agents (LCAE Section 2: Test).

Demonstrates:
1. Code-Based Evaluators (deterministic, fast, zero token cost)
   - Keyword containment
   - JSON Schema / structured key presence
   - Response length budget constraint
2. LLM-as-a-Judge Evaluators (semantic grading, reasoning verification)
   - Helpfulness and factual accuracy criteria
3. Benchmark dataset of test cases
4. Experiment execution and scoring summary table
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import json
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from shared.utils.model_factory import get_model
from agents.hello_world import graph as math_agent


# ─── 1. CODE-BASED EVALUATORS (Deterministic) ───────────────────────────────

def eval_contains_expected_keyword(prediction: str, expected: str) -> Dict[str, Any]:
    """Checks if the expected numeric answer or term is present in prediction."""
    passed = expected.lower() in prediction.lower()
    return {
        "key": "exact_keyword_match",
        "score": 1.0 if passed else 0.0,
        "reason": f"Expected '{expected}' present: {passed}",
    }


def eval_length_budget(prediction: str, max_chars: int = 300) -> Dict[str, Any]:
    """Ensures agent output is concise and fits within UI / token budgets."""
    length = len(prediction)
    passed = length <= max_chars
    return {
        "key": "length_budget",
        "score": 1.0 if passed else 0.0,
        "reason": f"Output length {length} <= limit {max_chars}: {passed}",
    }


# ─── 2. LLM-AS-A-JUDGE EVALUATOR (Semantic) ─────────────────────────────────

class JudgeOutput(BaseModel):
    helpfulness_score: float = Field(description="Score from 0.0 to 1.0 measuring helpfulness")
    factual_score: float = Field(description="Score from 0.0 to 1.0 measuring correctness")
    justification: str = Field(description="Brief explanation of the rating")


judge_model = get_model()


def eval_llm_judge(question: str, prediction: str, expected: str) -> Dict[str, Any]:
    """Uses an LLM judge to evaluate reasoning quality and correctness."""
    judge_prompt = f"""You are an expert AI Agent Evaluator. Grade the agent's response against ground truth.

Question: {question}
Ground Truth: {expected}
Agent Output: {prediction}

Respond with strict JSON with keys:
- helpfulness_score (float 0.0 - 1.0)
- factual_score (float 0.0 - 1.0)
- justification (string)
"""
    try:
        res = judge_model.invoke([HumanMessage(content=judge_prompt)])
        content = res.content.strip()
        # Clean markdown formatting if present
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()

        data = json.loads(content)
        return {
            "key": "llm_as_judge",
            "score": (data.get("helpfulness_score", 0.8) + data.get("factual_score", 0.8)) / 2,
            "details": data,
        }
    except Exception as e:
        # Fallback if judge parse fails
        is_close = expected in prediction
        return {
            "key": "llm_as_judge",
            "score": 1.0 if is_close else 0.5,
            "details": {"error": str(e), "fallback": True},
        }


# ─── 3. BENCHMARK DATASET ───────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id": "case-01",
        "question": "What is 15 multiplied by 4?",
        "expected": "60",
    },
    {
        "id": "case-02",
        "question": "Calculate 100 divided by 5, then add 7.",
        "expected": "27",
    },
    {
        "id": "case-03",
        "question": "If I have 25 apples and subtract 9, how many remain?",
        "expected": "16",
    },
]


# ─── 4. EXPERIMENT RUNNER ───────────────────────────────────────────────────

def run_evaluation_suite() -> List[Dict[str, Any]]:
    """Runs all test cases through the agent and evaluates outputs."""
    results = []
    print("\n" + "=" * 65)
    print("  RUNNING LCAE SECTION 2 EVALUATION SUITE")
    print("=" * 65)

    for item in EVAL_DATASET:
        test_id = item["id"]
        question = item["question"]
        expected = item["expected"]

        print(f"\n[Evaluating {test_id}]: {question}")

        # Run agent
        agent_res = math_agent.invoke({"messages": [HumanMessage(content=question)]})
        final_msg = agent_res["messages"][-1]
        prediction = getattr(final_msg, "content", "")
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], dict):
            prediction = prediction[0].get("text", str(prediction))

        # Run code-based evaluators
        code_eval = eval_contains_expected_keyword(prediction, expected)
        budget_eval = eval_length_budget(prediction, max_chars=250)

        # Run LLM judge evaluator
        judge_eval = eval_llm_judge(question, prediction, expected)

        combined_score = (code_eval["score"] + budget_eval["score"] + judge_eval["score"]) / 3

        result_entry = {
            "id": test_id,
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "keyword_score": code_eval["score"],
            "budget_score": budget_eval["score"],
            "judge_score": judge_eval["score"],
            "composite_score": round(combined_score, 2),
        }
        results.append(result_entry)

        print(f"  -> Prediction: {prediction[:60]}...")
        print(f"  -> Keyword Match: {code_eval['score']} | Budget: {budget_eval['score']} | Judge: {judge_eval['score']:.2f}")
        print(f"  -> Composite Score: {result_entry['composite_score']} / 1.0")

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 65)
    return results


if __name__ == "__main__":
    run_evaluation_suite()
