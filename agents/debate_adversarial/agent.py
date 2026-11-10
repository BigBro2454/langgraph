"""
Debate / Adversarial Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 5.3
"Two or more agents take opposing positions on a question and argue iteratively
until they converge or a judge agent makes a final determination. This pattern
leverages cognitive diversity to improve output quality, identifying flaws that
single-agent systems miss."

Pattern Card:
- Type: Multi-Agent, Dialectical / Adversarial Reasoning
- Best For: High-stakes architectural decisions, safety & compliance audits, policy evaluations
- Complexity: Medium
- Performance Lift: -30% to -50% reduction in hallucination rates; superior nuance and rigor
- Key Risk: 2-4x cost; latency scales linearly with debate rounds; risk of argumentative polarization
"""

from typing import List, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from shared.utils.model_factory import get_model


class DebateRound(TypedDict):
    proposer: str
    opponent: str


class DebateState(TypedDict):
    """Graph state tracking dialectical debate rounds and judge determination."""
    topic: str
    rounds: List[DebateRound]
    current_round: int
    max_rounds: int
    current_proposer_arg: Optional[str]
    judge_verdict: Optional[str]


model = get_model()

PROPOSER_PROMPT = """You are the Proponent Advocate Agent.
Your objective is to construct the strongest, most compelling, evidence-backed affirmative case
for the user's topic.
If an opponent has raised objections in previous rounds, directly address and refute their criticisms
while sharpening your core thesis.
"""

OPPONENT_PROMPT = """You are the Devil's Advocate & Adversarial Critic Agent.
Your objective is to stress-test the proponent's arguments. Uncompromisingly identify:
1. Hidden assumptions, logical fallacies, or unverified claims.
2. Catastrophic edge cases, failure scenarios, and operational risks.
3. Alternative perspectives and counter-evidence.
Be rigorous, precise, and intellectually demanding.
"""

JUDGE_PROMPT = """You are the Impartial Supreme Arbiter & Synthesizer.
Review the multi-round debate between the Proponent and the Adversarial Critic.
Evaluate:
1. Strength of empirical arguments vs speculative assertions.
2. Validity of counter-arguments raised by the Critic and how well they were addressed.
3. Actionable middle-ground or decisive verdict.

Provide a comprehensive, balanced ruling with:
- Summary of key contentions
- Strengths and weaknesses of each side
- Final authoritative determination / recommended action plan
"""


def proposer_node(state: DebateState) -> dict:
    """Proponent formulates initial position or defends against opponent critiques."""
    topic = state["topic"]
    rounds = state.get("rounds", [])
    current_round = state.get("current_round", 0) + 1

    history = []
    for idx, r in enumerate(rounds, 1):
        history.append(f"Round {idx} Proponent:\n{r['proposer']}")
        history.append(f"Round {idx} Opponent Counter:\n{r['opponent']}")

    prompt = (
        f"Debate Topic: {topic}\n\n"
        f"Debate History:\n" + ("\n\n".join(history) if history else "Opening Round") +
        f"\n\nDeliver your argument for Round {current_round}."
    )

    resp = model.invoke([
        SystemMessage(content=PROPOSER_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {
        "current_proposer_arg": str(resp.content),
        "current_round": current_round,
    }


def opponent_node(state: DebateState) -> dict:
    """Opponent critiques the latest proponent argument."""
    topic = state["topic"]
    proposer_arg = state["current_proposer_arg"]
    rounds = list(state.get("rounds", []))

    prompt = (
        f"Debate Topic: {topic}\n\n"
        f"Proponent's Latest Claim:\n{proposer_arg}\n\n"
        "Deliver your critical counter-argument and challenge any weak premises."
    )

    resp = model.invoke([
        SystemMessage(content=OPPONENT_PROMPT),
        HumanMessage(content=prompt),
    ])

    opponent_arg = str(resp.content)
    rounds.append({
        "proposer": proposer_arg,
        "opponent": opponent_arg,
    })

    return {
        "rounds": rounds,
        "current_proposer_arg": None,
    }


def judge_node(state: DebateState) -> dict:
    """Judge analyzes full debate history and renders a definitive consensus verdict."""
    topic = state["topic"]
    rounds = state.get("rounds", [])

    transcript = []
    for idx, r in enumerate(rounds, 1):
        transcript.append(
            f"=== ROUND {idx} ===\n"
            f"[PROPONENT]:\n{r['proposer']}\n\n"
            f"[OPPONENT CRITIC]:\n{r['opponent']}\n"
        )

    prompt = (
        f"Debate Topic: {topic}\n\n"
        f"Complete Debate Record:\n" + "\n".join(transcript) +
        "\nRender your final Supreme Arbiter verdict."
    )

    resp = model.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=prompt),
    ])

    return {"judge_verdict": str(resp.content)}


def should_continue_debate(state: DebateState) -> str:
    """Checks whether to proceed with another round or route to the judge."""
    if state.get("current_round", 0) >= state.get("max_rounds", 2):
        return "judge"
    return "proposer"


builder = StateGraph(DebateState)

builder.add_node("proposer", proposer_node)
builder.add_node("opponent", opponent_node)
builder.add_node("judge", judge_node)

builder.add_edge(START, "proposer")
builder.add_edge("proposer", "opponent")
builder.add_conditional_edges(
    "opponent",
    should_continue_debate,
    {
        "proposer": "proposer",
        "judge": "judge",
    },
)
builder.add_edge("judge", END)

graph = builder.compile()
