"""
CLI Runner for LangGraph Agents.

Usage:
    python runners/run_agent.py [agent_name] [query]

Examples:
    python runners/run_agent.py hello_world "What is 15 * 8?"
    python runners/run_agent.py persistent "My name is Priya and I build AI products"
    python runners/run_agent.py hitl "Deploy production release v2.4"
    python runners/run_agent.py supervisor "Synthesize findings on LangGraph state machines"
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import argparse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agents import (
    hello_world_agent,
    persistent_agent,
    hitl_agent,
    supervisor_agent,
    deep_agent,
)

console = Console()

AGENTS = {
    "hello_world": hello_world_agent,
    "persistent": persistent_agent,
    "hitl": hitl_agent,
    "supervisor": supervisor_agent,
    "deep_agent": deep_agent,
}


def run():
    parser = argparse.ArgumentParser(description="LangGraph Agent CLI Runner")
    parser.add_argument(
        "agent",
        choices=list(AGENTS.keys()),
        nargs="?",
        default="hello_world",
        help="The agent to execute (default: hello_world)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Calculate 45 * 12 + 10",
        help="Input query or prompt for the agent",
    )
    parser.add_argument(
        "--thread-id",
        default="cli-session-1",
        help="Thread ID for persistent sessions",
    )

    args = parser.parse_args()
    target_graph = AGENTS[args.agent]
    config = {"configurable": {"thread_id": args.thread_id}}

    console.print(
        Panel.fit(
            f"[bold cyan]LangGraph Runner[/bold cyan] | Agent: [bold green]{args.agent}[/bold green]\n"
            f"Input: [bold white]{args.query}[/bold white]\n"
            f"Thread ID: [dim]{args.thread_id}[/dim]",
            border_style="blue",
        )
    )

    input_payload = {"messages": [HumanMessage(content=args.query)]}
    result = target_graph.invoke(input_payload, config=config)

    # Check if agent was interrupted (HITL)
    if getattr(target_graph, "checkpointer", None):
        state = target_graph.get_state(config)
        if state.next:
            console.print(
                Panel(
                    f"[bold yellow]⚠️ AGENT PAUSED AT HUMAN APPROVAL GATE[/bold yellow]\n"
                    f"Waiting at node: [bold]{state.next}[/bold]\n"
                    f"Action proposal: {result.get('action_pending')}\n\n"
                    f"Auto-resuming with approval for demonstration...",
                    border_style="yellow",
                )
            )
            resumed = target_graph.invoke(Command(resume="APPROVED"), config=config)
            console.print(
                Panel(
                    f"[bold green]Approval Result:[/bold green]\n{resumed.get('execution_result')}",
                    border_style="green",
                )
            )
            return

    # Print final output
    messages = result.get("messages", [])
    if messages:
        final_msg = messages[-1]
        content = getattr(final_msg, "content", "")
        if isinstance(content, list) and content and isinstance(content[0], dict):
            content = content[0].get("text", str(content))

        console.print(
            Panel(
                f"[bold green]Agent Response:[/bold green]\n\n{content}",
                border_style="green",
            )
        )

    # If supervisor produced report_draft
    if result.get("report_draft"):
        console.print(
            Panel(
                f"[bold cyan]Final Synthesized Report:[/bold cyan]\n\n{result.get('report_draft')}",
                border_style="cyan",
            )
        )


if __name__ == "__main__":
    run()
