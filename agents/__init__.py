"""
LangGraph Agents Catalog.

Comprehensive implementations of production Agentic Design Patterns from
the 'Agentic Design Patterns Handbook' by Felix Williams (Book 4):

Single-Agent Patterns:
- react_agent: Interleaved reasoning and tool-calling loop (Ch 2.1)
- plan_and_execute_agent: Strategic planner + tactical executor + replanner (Ch 2.2)
- reflexion_agent: Dynamic verbal self-reflection + episodic memory retry (Ch 2.3)
- tree_of_thought_agent: Multi-branch exploration with BFS/beam search & pruning (Ch 2.4)
- mcts_agent: Monte Carlo Tree Search with UCB1 selection, expansion & rollout (Ch 2.5)

Multi-Agent Patterns:
- supervisor_agent: Hub-and-spoke central coordinator with specialist workers (Ch 5.1)
- hierarchical_teams_agent: Multi-tiered team structure with leads and workers (Ch 5.2)
- debate_adversarial_agent: Proponent vs Opponent with Judge arbitration (Ch 5.3)
- market_based_agent: Task broker auction with competitive agent bidding (Ch 5.4)
- sequential_pipeline_agent: Linear assembly-line stages with typed state contracts (Ch 5.5)
- parallel_execution_agent: Fan-out / Fan-in concurrent multi-perspective audits (Ch 5.6)

Reliability & Safety Patterns:
- hitl_agent: Human-in-the-loop approval gates via interrupt() and resume (Ch 8.3)
- guardrails_agent: 3-layer input screening, budget controls, and Constitutional AI (Ch 7.3 & 8.1)
- persistent_agent: MemorySaver state checkpointer and thread isolation (Ch 3.5 & 8.4)
- deep_agent: Long-running autonomous harness with context compression
- hello_world_agent: Foundational tool-calling ReAct graph
"""

from .hello_world.agent import graph as hello_world_agent
from .persistent_agent.agent import graph as persistent_agent
from .hitl_agent.agent import graph as hitl_agent
from .supervisor.agent import graph as supervisor_agent
from .deep_agent_pattern.agent import graph as deep_agent

# Book 4 Patterns
from .react_pattern.agent import graph as react_agent
from .plan_and_execute.agent import graph as plan_and_execute_agent
from .reflexion.agent import graph as reflexion_agent
from .tree_of_thought.agent import graph as tree_of_thought_agent
from .mcts.agent import graph as mcts_agent
from .hierarchical_teams.agent import graph as hierarchical_teams_agent
from .debate_adversarial.agent import graph as debate_adversarial_agent
from .market_based.agent import graph as market_based_agent
from .sequential_pipeline.agent import graph as sequential_pipeline_agent
from .parallel_execution.agent import graph as parallel_execution_agent
from .guardrails_constitutional.agent import graph as guardrails_agent

__all__ = [
    # Baseline
    "hello_world_agent",
    "persistent_agent",
    "hitl_agent",
    "supervisor_agent",
    "deep_agent",
    # Single-Agent Patterns (Book 4 Ch 2)
    "react_agent",
    "plan_and_execute_agent",
    "reflexion_agent",
    "tree_of_thought_agent",
    "mcts_agent",
    # Multi-Agent Patterns (Book 4 Ch 5)
    "hierarchical_teams_agent",
    "debate_adversarial_agent",
    "market_based_agent",
    "sequential_pipeline_agent",
    "parallel_execution_agent",
    # Reliability & Safety (Book 4 Ch 7 & 8)
    "guardrails_agent",
]
