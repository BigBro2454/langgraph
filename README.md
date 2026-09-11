# LangGraph & LangChain Agents

Production multi-agent workflows, state machines, and evaluation frameworks built with **LangGraph**, **LangChain**, and **LangSmith**.

This repository follows the **LangChain Certified Agent Engineer (LCAE)** curriculum, providing hands-on implementations for:
1. **Section 1: Build** — Graphs, Nodes, Conditional Edges, Tools, Middleware, and Sandboxing.
2. **Section 2: Test** — Code-based evaluators, LLM-as-judge, Offline/Online evals, and Datasets.
3. **Section 3: Deploy** — Deployments, Revisions, Assistants, Threads, and Storage Durability.
4. **Section 4: Monitor** — Traces, Thread analysis, Cost/Sentiment tracking, Online Evals vs. Insights, and Alerts.

---

## Quick Start

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Verify dependencies
python main.py

# 3. View interactive documentation
open docs/agents_documentation.html
```

---

## Agents Implemented

- `hello_world`: Foundational graph with custom math & greeting tools.
- `persistent_agent`: Stateful agent with checkpointers (`MemorySaver`/`SqliteSaver`), thread management, and time travel.
- `hitl_agent`: Human-in-the-loop agent with dynamic interrupts and approval gates.
- `supervisor`: Multi-agent team with coordinator routing to specialized sub-agents.
- `reliable_eval_agent`: Verification framework with deterministic and LLM-as-judge evaluators.
