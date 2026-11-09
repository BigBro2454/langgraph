"""
ReAct (Reason + Act) Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 2.1
"The ReAct pattern interleaves reasoning traces with tool-calling actions in a
single prompt loop. At each step, the agent generates a thought (reasoning about
what to do), takes an action (calls a tool), and observes the result before
deciding the next step."

Pattern Card:
- Type: Single-Agent, Linear Reasoning Loop
- Best For: Fact retrieval, arithmetic, straightforward multi-step questions (2-8 steps)
- Complexity: Low
- Risk: Error accumulation beyond 8-10 steps; infinite loops without turn limits
"""

from typing import Annotated, Sequence, TypedDict, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from shared.utils.model_factory import get_model
from shared.tools.calculator import add, subtract, multiply, divide
from shared.tools.web_search import search_knowledge_base


class ReActState(TypedDict):
    """State schema for ReAct Agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    thoughts: List[str]
    turn_count: int
    final_answer: Optional[str]


tools = [add, subtract, multiply, divide, search_knowledge_base]
tool_node = ToolNode(tools)

raw_model = get_model()
model_with_tools = raw_model.bind_tools(tools)

REACT_SYSTEM_PROMPT = """You are an autonomous ReAct (Reason + Act) agent.
Solve the user's task using the strict ReAct cycle:
1. THOUGHT: Reason step-by-step about what information is needed and what tool to call next.
2. ACTION: Call the required tool with explicit parameters.
3. OBSERVATION: Analyze the tool output.
4. FINAL ANSWER: When you have sufficient information, synthesize the complete answer.

Always state your reasoning clearly before calling any tool. If you have enough facts to answer, provide the final answer directly without calling more tools.
"""

MAX_TURNS = 6


def reason_node(state: ReActState) -> dict:
    """Agent node: generates thoughts and decides whether to act or conclude."""
    messages = list(state.get("messages", []))
    thoughts = list(state.get("thoughts", []))
    turn = state.get("turn_count", 0) + 1

    # Inject system instruction if not present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + messages

    response = model_with_tools.invoke(messages)

    # Extract reasoning thought if present
    content = response.content or ""
    if isinstance(content, list) and content and isinstance(content[0], dict):
        content_text = content[0].get("text", "")
    else:
        content_text = str(content)

    new_thoughts = thoughts
    if content_text.strip():
        new_thoughts = thoughts + [f"Turn {turn}: {content_text.strip()}"]

    final_answer = None
    # If no tool calls were made, this response is the final answer
    if not response.tool_calls:
        final_answer = content_text

    return {
        "messages": [response],
        "thoughts": new_thoughts,
        "turn_count": turn,
        "final_answer": final_answer,
    }


def should_continue(state: ReActState) -> str:
    """Conditional router: checks for tool calls, turn limits, or completion."""
    turn = state.get("turn_count", 0)
    messages = state.get("messages", [])

    if turn >= MAX_TURNS:
        return "max_turns_reached"

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

    return "conclude"


def handle_max_turns(state: ReActState) -> dict:
    """Safety fallback to prevent infinite loops (Williams Chapter 8.6)."""
    msg = AIMessage(
        content="[Circuit Breaker Triggered]: Reached maximum allowed reasoning turns without converging. "
                "Returning best-effort summary of gathered observations."
    )
    return {
        "messages": [msg],
        "final_answer": msg.content,
    }


builder = StateGraph(ReActState)

builder.add_node("reason", reason_node)
builder.add_node("tools", tool_node)
builder.add_node("handle_max_turns", handle_max_turns)

builder.add_edge(START, "reason")
builder.add_conditional_edges(
    "reason",
    should_continue,
    {
        "tools": "tools",
        "conclude": END,
        "max_turns_reached": "handle_max_turns",
    },
)
builder.add_edge("tools", "reason")
builder.add_edge("handle_max_turns", END)

graph = builder.compile()
