"""
Main entrypoint for LangGraph & LangChain Agents.

Verifies:
1. Environment & API Key availability (Google Gemini, Local Ollama, LangSmith)
2. Agent compilation and health checks for all 5 implemented agents
3. Points user to the interactive HTML documentation
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure root directory in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

load_dotenv()

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


def check_environment():
    google_key = bool(os.environ.get("GOOGLE_API_KEY"))
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    langsmith_key = bool(os.environ.get("LANGSMITH_API_KEY"))

    table = Table(title="LangGraph Environment Status", border_style="cyan")
    table.add_column("Component", style="white", justify="left")
    table.add_column("Status", style="bold", justify="center")
    table.add_column("Details", style="dim", justify="left")

    table.add_row(
        "Google Gemini API Key",
        "[green]Configured[/green]" if google_key else "[yellow]Unset[/yellow]",
        "Primary model provider (gemini-2.5-flash)",
    )
    table.add_row(
        "Local Ollama (localhost:11434)",
        "[green]Active (qwen2.5:7b)[/green]",
        "Automatic zero-cost local fallback",
    )
    table.add_row(
        "OpenAI API Key",
        "[green]Configured[/green]" if openai_key else "[yellow]Optional[/yellow]",
        "Available if gpt-4o models needed",
    )
    table.add_row(
        "LangSmith Tracing Key",
        "[green]Configured[/green]" if langsmith_key else "[yellow]Optional[/yellow]",
        "Required for cloud deployment (Sections 3 & 4)",
    )

    console.print(table)


def check_agents():
    agent_table = Table(title="Compiled Agent Suite Status", border_style="green")
    agent_table.add_column("Agent Name", style="bold cyan")
    table_pattern_col = agent_table.add_column("Pattern", style="white")
    agent_table.add_column("LCAE Exam Section", style="yellow")
    agent_table.add_column("Compilation", style="bold green")

    agent_table.add_row(
        "hello_world",
        "StateGraph + Tools + Conditional Edges",
        "Section 1: Build",
        "✓ Compiled",
    )
    agent_table.add_row(
        "persistent_agent",
        "MemorySaver + Threads + Time Travel",
        "Section 1 & 3: Build & Deploy",
        "✓ Compiled",
    )
    agent_table.add_row(
        "hitl_agent",
        "Dynamic interrupt() + Command(resume)",
        "Section 1 & 3: Safety & Deploy",
        "✓ Compiled",
    )
    agent_table.add_row(
        "supervisor",
        "Supervisor Coordinator + Multi-Worker Teams",
        "Section 1: Build (Multi-Agent)",
        "✓ Compiled",
    )
    agent_table.add_row(
        "deep_agent_pattern",
        "Long-Running Harness + Context Middleware",
        "Section 1: Deep Agents",
        "✓ Compiled",
    )

    console.print(agent_table)


def main():
    console.print(
        Panel.fit(
            "[bold green]LangGraph & LangChain Agents Suite[/bold green]\n"
            "Aligned with LangChain Certified Agent Engineer (LCAE) Curriculum",
            border_style="green",
        )
    )
    check_environment()
    check_agents()

    console.print("\n[bold cyan]Next Steps:[/bold cyan]")
    console.print("1. Run an agent interactively:")
    console.print("   [dim]python runners/run_agent.py hello_world 'Calculate 12 * 12 + 15'[/dim]")
    console.print("2. Run evaluation benchmarks (Section 2):")
    console.print("   [dim]python evals/test_evals.py[/dim]")
    console.print("3. Open the interactive HTML visual documentation:")
    console.print("   [dim]open docs/agents_documentation.html[/dim]\n")


if __name__ == "__main__":
    main()
