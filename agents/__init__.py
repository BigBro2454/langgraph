"""
LangGraph Agents Catalog.

Provides access to production agents implemented across the LCAE curriculum:
- hello_world: Foundational tool-calling ReAct graph
- persistent_agent: Stateful checkpointer & time travel memory
- hitl_agent: Human-in-the-loop dynamic interrupt & resumption
- supervisor: Multi-agent coordinator routing to specialist workers
- deep_agent_pattern: Long-running harness with context engineering
"""

from .hello_world.agent import graph as hello_world_agent
from .persistent_agent.agent import graph as persistent_agent
from .hitl_agent.agent import graph as hitl_agent
from .supervisor.agent import graph as supervisor_agent
from .deep_agent_pattern.agent import graph as deep_agent

__all__ = [
    "hello_world_agent",
    "persistent_agent",
    "hitl_agent",
    "supervisor_agent",
    "deep_agent",
]
