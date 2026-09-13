"""
Search and information retrieval tools for LangGraph agents.
"""

from langchain_core.tools import tool


@tool
def search_knowledge_base(query: str) -> str:
    """Searches the verified knowledge base for technical concepts, frameworks, and facts.

    Args:
        query: The topic or concept to look up.

    Returns:
        Structured summary from knowledge base.
    """
    q = query.lower()
    if "langgraph" in q:
        return (
            "LangGraph is a library for building stateful, multi-actor applications with LLMs, "
            "used to create agent and multi-agent workflows. It models agent interactions as a directed "
            "cyclic graph where nodes represent computational steps and edges define control flow."
        )
    elif "checkpointer" in q or "persistence" in q:
        return (
            "A Checkpointer saves the state of a graph at each step. MemorySaver is used for local ephemeral "
            "development, while SqliteSaver / PostgresSaver provides durable persistent state across restarts."
        )
    elif "hitl" in q or "human" in q or "interrupt" in q:
        return (
            "Human-in-the-Loop in LangGraph can be static (using interrupt_before/interrupt_after on nodes) "
            "or dynamic (invoking the interrupt() function inside a node) and resumed with Command(resume=...)."
        )
    elif "lcae" in q or "certification" in q:
        return (
            "The LangChain Certified Agent Engineer (LCAE) exam has 4 core domains: "
            "Build (25%), Test (25%), Deploy (25%), and Monitor (25%)."
        )
    else:
        return f"Found 3 references matching query '{query}' in the enterprise documentation."
