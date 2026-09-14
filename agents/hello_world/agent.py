"""
hello_world — Foundational LangGraph Agent.

Demonstrates:
- StateGraph definition with TypedDict and message reducers (add_messages)
- Binding tools to a chat model (Google Gemini with Ollama fallback)
- ToolNode execution and tools_condition routing
- Basic cyclic reasoning loop: START -> Agent -> (tools -> Agent)* -> END
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from shared.utils.model_factory import get_model
from shared.tools.calculator import add, subtract, multiply, divide


class AgentState(TypedDict):
    """The graph state schema.

    'messages' holds the conversation history. The add_messages reducer ensures
    that returning a list of new messages appends them rather than overwriting.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define tools
tools = [add, subtract, multiply, divide]
tool_node = ToolNode(tools)

# Initialize model and bind tools
raw_model = get_model()
model_with_tools = raw_model.bind_tools(tools)

SYSTEM_INSTRUCTION = (
    "You are a helpful math and reasoning assistant built with LangGraph. "
    "When asked arithmetic or calculation questions, always call the appropriate tools "
    "to compute accurate answers. Be concise and friendly."
)


def call_model(state: AgentState) -> dict:
    """Invokes the model with the current conversation history."""
    messages = list(state["messages"])
    # Prepend system instruction if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_INSTRUCTION)] + messages

    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# Build the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add edges
workflow.add_edge(START, "agent")
# tools_condition inspects the last message; routes to 'tools' if tool_calls present, else END
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

from langgraph.checkpoint.memory import MemorySaver

# Add MemorySaver checkpointer for thread isolation
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
