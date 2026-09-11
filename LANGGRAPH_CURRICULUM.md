# LangGraph & LangChain — Learning Curriculum
### From Zero to Certified Agent Engineer

> **Target Certification:** LangChain Certified Agent Engineer (LCAE)
> **Your Repo:** github.com/BigBro2454 | **LangGraph Version:** 0.4+ (2026) | **Model:** GPT-4o / Claude / Gemini
> **Exam PDF:** [exam_guide.pdf](file:///Users/ishan03/Workspace/projects/ai/langgraph/exam_guide.pdf)

---

## Learning Path Overview

```
Phase 1  Foundations          (Weeks 1-3)   Beginner
Phase 2  Agent Development    (Weeks 4-6)   Intermediate
Phase 3  Testing & Evaluation (Weeks 7-8)   Advanced
Phase 4  Deploy & Monitor     (Weeks 9-12)  Expert / Exam-Ready
```

---

## LCAE Exam Structure

The certification is a **2-hour proctored exam** with **40 multiple-choice questions** split evenly across four sections. Understanding this structure shapes the entire curriculum.

| Section | Questions | Weight | Course Prerequisite |
|---|---|---|---|
| 1 · Build | 10 | 25% | [LangChain Foundations](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python) · [Deep Agents Foundations](https://academy.langchain.com/courses/foundation-introduction-to-deepagents) |
| 2 · Test | 10 | 25% | [Building Reliable Agents](https://academy.langchain.com/courses/building-reliable-agents) |
| 3 · Deploy | 10 | 25% | [LangSmith Deployment Foundations](https://academy.langchain.com/courses/langsmith-deployment) |
| 4 · Monitor | 10 | 25% | [Monitoring Production Agents](https://academy.langchain.com/courses/production-monitoring) |

> **Semi-Open-Book:** You may use the whitelisted resources below during the exam. No search engines, AI assistants, or second screens allowed.

#### Allowed Resources During Exam

| Resource | URL |
|---|---|
| ✅ LangSmith | https://smith.langchain.com |
| ✅ LangChain Docs | https://docs.langchain.com |
| ✅ LangChain Academy | https://academy.langchain.com |

> ⛔ **NOT Allowed:** Google search, ChatGPT / Claude / AI assistants, the AI helper on docs.langchain.com, browser agents, second screens, outside communication, notes or scrap paper.

---

## Prerequisites

- [ ] Python 3.11+ installed
- [ ] `langchain`, `langgraph`, `langsmith` installed via pip/uv
- [ ] GitHub account connected — BigBro2454
- [ ] OpenAI / Anthropic / Google API key configured
- [ ] LangSmith account → https://smith.langchain.com
- [ ] LangChain Academy account → https://academy.langchain.com
- [ ] Basic Python knowledge (functions, classes, type hints, TypedDict)

---

## What's Current in LangGraph/LangChain as of September 2026

The curriculum below is fully up to date. Here is what changed in 2026 that you should know before starting:

### create_react_agent → create_agent Migration

The biggest API shift in 2026. The legacy `create_react_agent` from `langgraph.prebuilt`
has been **deprecated** in favor of the new `create_agent` from `langchain.agents`.

What this means for you:
- The old ReAct-specific loop is replaced by a more flexible, middleware-aware agent factory
- `create_agent` is built on top of the LangGraph runtime but offers a cleaner interface
- You can now inject middleware for cross-cutting concerns (logging, guardrails, context pruning)
- Import path changed: `from langchain.agents import create_agent`

### Deep Agents — Long-Running Agent Harness

LangChain introduced **Deep Agents** (`deepagents`) as an open-source agent harness
for building long-running, complex agents that go beyond simple tool-calling loops.

| Agent Type | Description | Use When |
|---|---|---|
| `create_agent` | Standard tool-calling agent with middleware | Most common use cases |
| `deepagents` | Long-running agent harness with durable execution | Complex, multi-step workflows |

### Context Engineering — AGENTS.md & SKILL.md

The 2026 ecosystem standardized on file-based context management:

| File | Purpose | Loaded When |
|---|---|---|
| `AGENTS.md` | Global project rules, conventions, architecture | Always at startup |
| `SKILL.md` | Task-specific capability with YAML frontmatter | On-demand via progressive disclosure |

### Middleware Architecture

Middleware provides hooks into the agent execution lifecycle:

```
before_model       → modify the prompt, inject context, prune history
modify_model_request → transform the request before it hits the LLM
after_model        → validate output, apply guardrails, log responses
```

Built-in middleware includes `SummarizationMiddleware`, `HumanInTheLoopMiddleware`,
and you can write custom middleware for any cross-cutting concern.

### LangSmith Deployment Platform (GA)

LangSmith is now a full deployment platform, not just observability. Key primitives:

| Concept | Description |
|---|---|
| Graph | Your compiled LangGraph agent |
| Deployment | A running instance of your graph |
| Revision | A versioned update to a deployment |
| Assistant | A configured instance with specific params |
| Thread | A persistent conversation/session |
| Run | A single execution within a thread |

---

## Phase 1 — Foundations
> Goal: Understand the LangGraph mental model and build your first stateful agent.

### Week 1 — What is LangGraph?

#### Concept: The State Machine Mental Model

LangGraph is NOT just LangChain with graphs. It is a **state machine framework** for
building agents with precise control over execution flow:

```
User Input
    |
State Initialization  (define what the agent remembers)
    |
Node Execution        (LLM call, tool use, or plain function)
    |
State Update          (node writes back to shared state)
    |
Edge Routing          (conditional or unconditional → next node)
    |
Cycle or End          (loop back for retry, or finish)
```

The key difference from vanilla LangChain: **LangGraph supports cycles, persistence,
and human-in-the-loop natively.**

#### Topics to Study

- [ ] What is LangGraph vs. LangChain vs. CrewAI vs. Google ADK?
- [ ] The three building blocks: State, Nodes, Edges
- [ ] Why cycles matter — retry loops, self-correction, reflection
- [ ] LangGraph's relationship to LangChain (runtime layer, not replacement)
- [ ] The Agent Development Lifecycle (ADLC): Build → Test → Deploy → Monitor

#### Resources

- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- LangChain Academy — [LangChain Foundations](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)
- DeepLearning.AI — AI Agents in LangGraph: https://www.deeplearning.ai

---

### Week 2 — State, Nodes, and Edges

#### Concept: State = The Agent's Shared Memory

Everything in LangGraph flows through state. Every node reads from it and writes to it.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # add_messages appends instead of overwriting
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int
    final_answer: str
```

> IMPORTANT: Use `Annotated` with reducer functions (like `add_messages` or
> `operator.add`) to control how state updates are merged. Without reducers,
> new values silently overwrite old ones.

#### Concept: Nodes = Functions That Transform State

```python
def research_node(state: AgentState) -> dict:
    """Calls the LLM to research the user's question."""
    response = model.invoke(state["messages"])
    return {"messages": [response], "step_count": state["step_count"] + 1}
```

#### Concept: Edges = Routing Logic

```python
# Conditional edge — route based on state
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"      # go to tool node
    return "end"            # finish

graph.add_conditional_edges("agent", should_continue, {
    "tools": "tool_node",
    "end": END,
})
```

#### Topics to Study

- [ ] Defining state schemas with TypedDict
- [ ] Reducer functions: `add_messages`, `operator.add`, custom reducers
- [ ] Writing node functions that read and return state updates
- [ ] Unconditional edges vs. conditional edges
- [ ] The `START` and `END` special nodes
- [ ] Compiling a graph with `graph.compile()`

#### Hands-On Projects

- [ ] Project 1.1 — Build a simple chatbot graph (single LLM node, no tools)
- [ ] Project 1.2 — Add a conditional edge: if user says "quit", route to END
- [ ] Project 1.3 — Build a graph with 3 nodes: greet → process → respond

---

### Week 3 — Tools and Your First ReAct Agent

#### Concept: Tool Calling in LangGraph

Tools in LangGraph follow the LangChain tool interface. The LLM decides when to
call a tool, and LangGraph orchestrates the execution:

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information.

    Args:
        query: The search term to look up.
    """
    return f"Results for: {query}"
```

#### Concept: The create_agent Factory (2026)

The modern way to build a standard tool-calling agent:

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(model=model, tools=[search_web])

# Run it
result = agent.invoke({"messages": [("user", "What's the weather?")]})
```

#### Topics to Study

- [ ] The `@tool` decorator and structured tool definitions
- [ ] `create_agent` vs. building a custom graph (when to use each)
- [ ] The `ToolNode` prebuilt for automatic tool execution
- [ ] How `create_agent` differs from the deprecated `create_react_agent`
- [ ] Binding tools to models with `model.bind_tools()`

#### Hands-On Projects

- [ ] Project 2.1 — Build a calculator agent with add/subtract/multiply tools
- [ ] Project 2.2 — Build a research agent using `create_agent` with a search tool
- [ ] Project 2.3 — Build the same agent as a custom graph (manually define nodes/edges)

#### Resources

- LangChain Academy — [LangChain Foundations](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)
- LangGraph How-To Guides: https://langchain-ai.github.io/langgraph/how-tos/

---

## Phase 2 — Agent Development (Exam Section 1: Build)
> Goal: Master production-grade agent patterns — middleware, context engineering,
> multi-agent systems, and human-in-the-loop.
>
> 📖 **Deep Dive Study Guide:** [chapters/01-build.md](file:///Users/ishan03/Workspace/projects/ai/langgraph/chapters/01-build.md)

### Week 4 — Persistence, Memory, and Checkpointers

#### Concept: Checkpointers = Agent's Durable Memory

Without a checkpointer, your agent forgets everything between invocations.
With one, it maintains full conversation history and can recover from crashes.

```python
from langgraph.checkpoint.memory import MemorySaver

# In-memory (development)
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# Every invocation needs a thread_id
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke({"messages": [("user", "Hello!")]}, config)
```

#### Checkpointer Options

| Checkpointer | Best For | Persistence |
|---|---|---|
| `MemorySaver` | Local development and testing | In-memory (lost on restart) |
| `SqliteSaver` | Single-server prototypes | File-based SQLite |
| `PostgresSaver` | Production deployments | Durable database |
| LangSmith Platform | Managed production | Fully managed by LangChain |

#### Concept: Threads = Persistent Conversations

A `thread_id` is the key to persistence. Same thread_id = continued conversation.
Different thread_id = fresh conversation.

```python
# Turn 1
graph.invoke({"messages": [("user", "My name is Ishan")]}, config)

# Turn 2 — agent remembers the name
graph.invoke({"messages": [("user", "What's my name?")]}, config)
# → "Your name is Ishan"
```

#### Topics to Study

- [ ] MemorySaver for development, PostgresSaver for production
- [ ] Thread IDs and their role in conversation persistence
- [ ] How checkpointers enable time-travel debugging (rewinding state)
- [ ] State snapshots and get_state() / update_state()
- [ ] The difference between thread-level memory and cross-thread memory

#### Hands-On Projects

- [ ] Project 3.1 — Add MemorySaver to your chatbot and have multi-turn conversations
- [ ] Project 3.2 — Use get_state() to inspect the agent's state at any checkpoint
- [ ] Project 3.3 — Implement time-travel: rewind to a previous state and fork the conversation

---

### Week 5 — Human-in-the-Loop, Middleware, and Context Engineering

#### Concept: Human-in-the-Loop (HITL)

LangGraph supports pausing agent execution at critical decision points for
human review before proceeding.

**Static Interrupts** — defined at compile time:

```python
# Pause BEFORE the tool node runs — human must approve
graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["tool_node"]
)
```

**Dynamic Interrupts** — called inside any node:

```python
from langgraph.types import interrupt, Command

def sensitive_action_node(state):
    # Pause and surface a question to the human
    approval = interrupt("About to delete user data. Approve? (yes/no)")
    if approval == "yes":
        return {"result": "deleted"}
    return {"result": "cancelled"}

# Resume with: Command(resume="yes")
```

#### Concept: Middleware (2026 — Exam Focus Area)

Middleware intercepts the agent's execution flow for cross-cutting concerns:

```python
from langchain.agents import create_agent
from langchain.middleware import SummarizationMiddleware

agent = create_agent(
    model=model,
    tools=tools,
    middleware=[
        SummarizationMiddleware(max_messages=20),  # auto-prune long contexts
    ]
)
```

| Middleware | Purpose |
|---|---|
| `SummarizationMiddleware` | Auto-summarize and prune long conversation history |
| `HumanInTheLoopMiddleware` | Inject approval gates before critical actions |
| Custom middleware | Logging, PII redaction, rate limiting, caching |

#### Concept: Context Engineering — AGENTS.md & SKILL.md (Exam Focus Area)

For long-running agents, managing context is critical:

```
AGENTS.md (Global Rules)
├── Project conventions, architecture decisions
├── Coding style, mandatory behaviors
└── Loaded ALWAYS at agent startup

SKILL.md (Task-Specific Capability)
├── YAML frontmatter: name, description
├── Full instructions loaded ON-DEMAND only
└── Progressive disclosure — saves token budget
```

#### Topics to Study

- [ ] Static interrupts: `interrupt_before`, `interrupt_after`
- [ ] Dynamic interrupts: `interrupt()` function and `Command(resume=...)`
- [ ] Writing custom middleware with `before_model` and `after_model` hooks
- [ ] Built-in middleware: SummarizationMiddleware
- [ ] AGENTS.md — structure and content for global project context
- [ ] SKILL.md — YAML frontmatter, progressive disclosure pattern
- [ ] Sandboxing agent execution (security considerations)

#### Hands-On Projects

- [ ] Project 4.1 — Add `interrupt_before` to your tool-calling agent for human approval
- [ ] Project 4.2 — Write a custom logging middleware that logs every LLM call
- [ ] Project 4.3 — Create an AGENTS.md for your project and 2 SKILL.md files

#### Resources

- LangChain Academy — [Deep Agents Foundations](https://academy.langchain.com/courses/foundation-introduction-to-deepagents)
- LangGraph HITL Guide: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/

---

### Week 6 — Multi-Agent Systems and Deep Agents

#### Concept: Multi-Agent Architectures

LangGraph supports building teams of specialized agents that collaborate:

```
Supervisor Agent (Manager)
    |
    +-- routes to → Research Agent  (searches for information)
    |
    +-- routes to → Writer Agent    (drafts content)
    |
    +-- routes to → Reviewer Agent  (validates quality)
    |
    +-- decides  → Task complete, return final output
```

#### Multi-Agent Patterns

| Pattern | Description | Best For |
|---|---|---|
| Supervisor | Manager routes tasks to specialist workers | Complex multi-step workflows |
| Swarm / Handoff | Agents hand off control to each other | Customer service, routing |
| Parallel Fan-Out | Multiple agents run simultaneously | Research across multiple sources |
| Hierarchical | Nested teams of supervisors and workers | Enterprise-scale systems |

#### Concept: Deep Agents (2026 — Exam Focus Area)

Deep Agents are LangChain's harness for long-running, complex agents:

```python
from langchain.agents import create_deep_agent

agent = create_deep_agent(
    model=model,
    tools=tools,
    # Deep agents handle durable execution, context management,
    # and complex multi-step workflows out of the box
)
```

Key differences from `create_agent`:
- Designed for tasks that take minutes to hours, not seconds
- Built-in durable execution (survives restarts)
- Automatic context management for very long conversations
- Native support for AGENTS.md and SKILL.md conventions

#### Topics to Study

- [ ] Supervisor pattern — one agent routes to many specialists
- [ ] Swarm / handoff pattern — agents transfer control dynamically
- [ ] `create_agent` vs. `create_deep_agent` — when to use each (exam focus)
- [ ] Subgraph composition — nesting one graph inside another
- [ ] State sharing between agents in multi-agent systems
- [ ] Building agent teams with specialized roles

#### Hands-On Projects

- [ ] Project 5.1 — Build a supervisor agent that routes to Research + Writer agents
- [ ] Project 5.2 — Build a swarm of 3 agents that hand off based on user intent
- [ ] Project 5.3 — Implement a parallel fan-out: query 3 data sources, merge results

#### Resources

- LangGraph Multi-Agent Tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/
- LangChain Academy — [Deep Agents Foundations](https://academy.langchain.com/courses/foundation-introduction-to-deepagents)

---

## Phase 3 — Testing & Evaluation (Exam Section 2: Test)
> Goal: Build reliable agents with proper evaluation, experimentation, and quality assurance.
>
> 📖 **Deep Dive Study Guide:** [chapters/02-test.md](file:///Users/ishan03/Workspace/projects/ai/langgraph/chapters/02-test.md)

### Week 7 — Evaluation Fundamentals

#### Concept: How Do You Know Your Agent Is Good?

Agents are non-deterministic — same input can produce different outputs each run.
LangSmith provides the evaluation infrastructure to measure quality systematically.

#### Two Types of Evaluators (Exam Focus Area)

```
Code-Based Evaluators          LLM-as-Judge Evaluators
├── Deterministic              ├── Non-deterministic
├── Fast and cheap             ├── Slower and costs LLM tokens
├── Exact match, regex, JSON   ├── Semantic similarity, correctness
├── Great for structure        ├── Great for quality/reasoning
└── e.g., "Did it return JSON?"└── e.g., "Is this answer helpful?"
```

#### Concept: Datasets and Experiments

```python
from langsmith import Client

client = Client()

# Create a dataset of test cases
dataset = client.create_dataset("agent-eval-v1")

# Add examples
client.create_example(
    dataset_id=dataset.id,
    inputs={"question": "What is the capital of France?"},
    outputs={"answer": "Paris"},
)

# Run an experiment — evaluates your agent against the dataset
results = client.run_on_dataset(
    dataset_name="agent-eval-v1",
    llm_or_chain_factory=my_agent,
    evaluation=my_evaluators,
)
```

#### Topics to Study

- [ ] Code-based evaluators: exact match, regex, JSON schema validation
- [ ] LLM-as-judge evaluators: correctness, helpfulness, faithfulness
- [ ] When to use code-based vs. LLM-as-judge (exam focus)
- [ ] Creating datasets in LangSmith (inputs + expected outputs)
- [ ] Running experiments: comparing agent versions against a dataset
- [ ] Interpreting experiment results without overfitting to noise
- [ ] Evaluator alignment — making sure your evaluator measures what matters

#### Hands-On Projects

- [ ] Project 6.1 — Create a dataset of 15 test cases for your chatbot agent
- [ ] Project 6.2 — Write 2 code-based evaluators (JSON check, keyword presence)
- [ ] Project 6.3 — Write 1 LLM-as-judge evaluator for answer quality

#### Resources

- LangChain Academy — [Building Reliable Agents](https://academy.langchain.com/courses/building-reliable-agents)
- LangSmith Evaluation Docs: https://docs.smith.langchain.com

---

### Week 8 — Advanced Evaluation and Reliability

#### Concept: Online vs. Offline Evaluation (Exam Focus Area)

```
Offline Evaluation                    Online Evaluation
├── Run BEFORE deployment             ├── Run AFTER deployment on live traffic
├── Uses pre-built datasets           ├── Evaluates real user interactions
├── "Lab testing"                     ├── "Production monitoring"
├── You control the inputs            ├── Inputs are unpredictable
└── Answers: "Is my agent ready?"     └── Answers: "Is my agent performing well?"
```

#### Concept: Adding Examples to a Dataset (Exam Focus Area)

You can grow your eval dataset from multiple sources:

| Source | Method |
|---|---|
| Manual creation | Write input/output pairs by hand |
| Production traces | Promote real user interactions from LangSmith traces |
| Failure analysis | Add cases from production errors to prevent regressions |
| Edge cases | Specifically craft tricky inputs |

#### Topics to Study

- [ ] Online evaluators — grading live production traffic in real-time
- [ ] Offline evaluators — pre-deployment quality gates
- [ ] Adding examples to datasets from traces, manual input, and production errors
- [ ] A/B testing different model versions, prompts, and tool configurations
- [ ] Avoiding overfitting: evaluation best practices
- [ ] Evaluator alignment techniques

#### Hands-On Projects

- [ ] Project 7.1 — Run a full experiment comparing GPT-4o vs. Claude on your agent
- [ ] Project 7.2 — Set up an online evaluator to grade live agent interactions
- [ ] Project 7.3 — Build a regression test suite from production failure cases

---

## Phase 4 — Deploy & Monitor (Exam Sections 3 & 4)
> Goal: Ship your agents to production with LangSmith, then monitor and maintain them.
>
> 📖 **Deep Dive Study Guides:** [chapters/03-deploy.md](file:///Users/ishan03/Workspace/projects/ai/langgraph/chapters/03-deploy.md) · [chapters/04-monitor.md](file:///Users/ishan03/Workspace/projects/ai/langgraph/chapters/04-monitor.md)

### Week 9 — Deployment Fundamentals

#### Concept: LangSmith Deployment Terminology (Exam Focus Area)

Understanding these six terms is critical for the exam:

```
Graph         Your compiled LangGraph agent code
    |
Deployment    A running instance of a Graph on LangSmith infrastructure
    |
Revision      A versioned update to a Deployment (like a git commit)
    |
Assistant     A configured instance — same graph, different parameters
    |
Thread        A persistent conversation or session (identified by thread_id)
    |
Run           A single execution within a Thread
```

#### Deployment with langgraph CLI

```bash
# langgraph.json — deployment configuration file
{
  "dependencies": ["./my_agent"],
  "graphs": {
    "my_agent": "./my_agent/graph.py:graph"
  }
}

# Deploy to LangSmith
langgraph deploy
```

#### Topics to Study

- [ ] The 6 deployment primitives: graph, deployment, revision, assistant, thread, run
- [ ] `langgraph.json` configuration file structure
- [ ] Deploying with `langgraph deploy` CLI
- [ ] Creating and managing revisions (versioning deployments)
- [ ] The Deployment UI in LangSmith
- [ ] Environment variables and secret handling in deployments

#### Hands-On Projects

- [ ] Project 8.1 — Write a `langgraph.json` for your agent and deploy locally
- [ ] Project 8.2 — Deploy your agent to LangSmith and create 2 revisions

#### Resources

- LangChain Academy — [LangSmith Deployment Foundations](https://academy.langchain.com/courses/langsmith-deployment)
- LangGraph Deploy Docs: https://langchain-ai.github.io/langgraph/cloud/

---

### Week 10 — Advanced Deployment Patterns

#### Concept: Multi-Region Serving (Exam Focus Area)

LangSmith supports three hosting models for different data residency and
availability requirements:

| Model | Control Plane | Data Plane | Best For |
|---|---|---|---|
| Cloud (Managed) | LangChain | LangChain | Fastest start, lowest ops burden |
| Hybrid | LangChain | Self-hosted | Data residency requirements |
| Self-Hosted | Self-hosted | Self-hosted | Full control, air-gapped environments |

#### Concept: Authentication vs. Authorization (Exam Focus Area)

```
Authentication (AuthN)          Authorization (AuthZ)
├── "WHO are you?"              ├── "WHAT can you do?"
├── API keys, OAuth, SSO        ├── Roles, permissions, scopes
└── Verified at the gateway     └── Checked per operation
```

#### Concept: Storage Durability

LangSmith deployments provide durable storage for:
- Checkpoints (up to 25 MB per payload)
- Memory stores for cross-thread data
- Thread history for conversation recovery

#### Topics to Study

- [ ] Multi-region deployment strategies
- [ ] Cloud vs. Hybrid vs. Self-hosted hosting models
- [ ] Authentication patterns: API keys, OAuth, SSO
- [ ] Authorization: role-based access control for agents
- [ ] Storage durability — checkpoints and memory stores
- [ ] Secret management in production deployments
- [ ] Multi-tenant architecture for serving multiple users

#### Hands-On Projects

- [ ] Project 9.1 — Configure authentication for your deployed agent
- [ ] Project 9.2 — Set up environment variables and secrets in LangSmith

---

### Week 11 — Monitoring and Observability

#### Concept: Reading Traces (Exam Focus Area)

Traces are the primary debugging tool in LangSmith. Every agent invocation
produces a trace showing the full execution path:

```
Trace (top-level)
    |
    +-- LLM Call (model, prompt, response, tokens, latency)
    |
    +-- Tool Call (input, output, duration)
    |
    +-- LLM Call (with tool result, final response)
    |
    +-- Metadata (cost, total tokens, total time)
```

#### Concept: Grouping Traces into Threads (Exam Focus Area)

Threads group related traces from multi-turn conversations:

```
Thread "user-123"
    |
    +-- Trace 1 (Turn 1: "What's the weather?")
    |
    +-- Trace 2 (Turn 2: "What about tomorrow?")
    |
    +-- Trace 3 (Turn 3: "Thanks!")
```

#### Concept: Online Evals vs. Insights (Exam Focus Area)

| Feature | Online Evals | Insights |
|---|---|---|
| What it does | Grades live production traces | Auto-clusters and analyzes trace patterns |
| When it runs | Real-time on every trace (or sampled) | Periodically over accumulated data |
| Output | Per-trace quality scores | Trends, patterns, anomaly detection |
| Best for | Continuous quality monitoring | Understanding agent behavior at scale |

#### Concept: Alerting (Exam Focus Area)

LangSmith supports project-scoped alerts for:
- **Error rate** spikes (agent failures)
- **Latency** increases (slow responses)
- **Cost** thresholds (unexpected LLM spending)
- **Feedback scores** dropping (user satisfaction)

Alerts can be routed via webhooks to Slack, PagerDuty, or custom endpoints.

#### Topics to Study

- [ ] Reading and navigating traces in LangSmith
- [ ] Understanding trace hierarchy: runs, spans, and nested calls
- [ ] Grouping traces into threads for conversation analysis
- [ ] Tracking costs per model, per agent, per user
- [ ] User sentiment tracking via feedback scores
- [ ] Online evaluations — grading production traces
- [ ] Insights — automated pattern discovery and clustering
- [ ] Setting up alerts for error rate, latency, cost, and quality

#### Hands-On Projects

- [ ] Project 10.1 — Deploy an agent and analyze its traces in LangSmith
- [ ] Project 10.2 — Set up an online evaluator on production traces
- [ ] Project 10.3 — Configure alerts for error rate and cost thresholds

#### Resources

- LangChain Academy — [Monitoring Production Agents](https://academy.langchain.com/courses/production-monitoring)
- LangSmith Docs: https://docs.smith.langchain.com

---

### Week 12 — Capstone Project & Exam Prep

Apply everything from Phases 1-4. Pick one option below, then take the practice exam.

---

**Option A — Personal Research Assistant**
- Architecture: Supervisor → Search Agent + Summarizer Agent + Writer Agent
- Features: Persistent threads, human approval before sending reports, middleware for logging
- Deployment: LangSmith managed deployment
- Monitoring: Full traces, online evals, cost tracking alerts
- Great for demonstrating all LCAE domains in one project.

---

**Option B — Customer Support Bot**
- Architecture: Swarm pattern with handoff between Triage, Billing, and Technical agents
- Features: Multi-turn thread persistence, context engineering with AGENTS.md
- Deployment: LangSmith with authentication and multi-tenant support
- Monitoring: Sentiment tracking, latency alerts, insights for common failure patterns

---

**Option C — AI PM Toolkit Agent**
*Connects to your pm-learning/ project!*
- Architecture: Deep Agent with SKILL.md files for Market Research, Strategy, and Communication
- Features: Long-running execution, progressive disclosure, middleware for PII redaction
- Deployment: LangSmith with secret management for API keys
- Monitoring: Online evals for output quality, cost alerts per skill

---

## Exam Preparation Checklist

### Practice Exam
- [ ] Complete the official [LCAE Practice Exam](https://academy.langchain.com/courses/lcae-practice-exam) on LangChain Academy
- [ ] Review all incorrect answers and study the related course material

### Day-Before Checklist
- [ ] Verify your machine meets [Talview system requirements](https://certification.talview.com/support/solutions/articles/11000130864-pre-requisites-for-online-test)
- [ ] Familiarize yourself with [docs.langchain.com](https://docs.langchain.com) navigation
- [ ] Familiarize yourself with [smith.langchain.com](https://smith.langchain.com) interface
- [ ] Have valid ID ready (passport, driver's licence, work ID, or national ID)
- [ ] Review all focus areas from each section (listed below)

### Section 1 — Build Focus Areas
- [ ] `create_agent` vs. `deepagents` — differences and when to use each
- [ ] Middleware — how to write and apply custom middleware
- [ ] Context engineering for long-running agents
- [ ] AGENTS.md and SKILL.md — structure and purpose
- [ ] Sandboxing — security considerations for agent execution

### Section 2 — Test Focus Areas
- [ ] Code-based evaluators vs. LLM-as-judge
- [ ] Online vs. offline evaluators
- [ ] Running and interpreting experiments
- [ ] Evaluator alignment — ensuring evaluators measure what matters
- [ ] Adding examples to a dataset (manual, from traces, from failures)

### Section 3 — Deploy Focus Areas
- [ ] Deployment terminology: graph, deployment, revision, assistant, thread, run
- [ ] Multi-region serving strategies
- [ ] Secret handling in deployments
- [ ] Authentication vs. authorization
- [ ] Storage durability — checkpoints and memory stores

### Section 4 — Monitor Focus Areas
- [ ] Reading traces and understanding trace hierarchy
- [ ] Grouping traces into threads
- [ ] Tracking costs and user sentiment
- [ ] Online Evals vs. Insights — differences and use cases
- [ ] Alerting — configuring and responding to alerts

---

## Reference Card

### LangGraph Core Classes

| Class / Concept | Purpose |
|---|---|
| `StateGraph` | Define a graph with typed state |
| `TypedDict` + `Annotated` | Define the state schema with reducers |
| `add_conditional_edges` | Route between nodes based on state |
| `MemorySaver` / `PostgresSaver` | Persist state across invocations |
| `interrupt()` / `Command` | Human-in-the-loop control flow |
| `ToolNode` | Pre-built node for automatic tool execution |

### LangChain Agent APIs (2026)

| API | Purpose |
|---|---|
| `create_agent` | Standard tool-calling agent with middleware support |
| `create_deep_agent` | Long-running agent harness for complex workflows |
| `@tool` | Decorator to define tools with schemas |
| Middleware | Before/after hooks for cross-cutting concerns |

### LangSmith Components

| Component | Purpose |
|---|---|
| Traces | Full execution logs for every agent invocation |
| Threads | Group related traces into conversations |
| Datasets | Collections of input/output pairs for evaluation |
| Experiments | Run agent against dataset with evaluators |
| Online Evals | Grade live production traffic |
| Insights | Auto-analyze and cluster trace patterns |
| Alerts | Notifications for error rate, latency, cost, quality |

### CLI Reference

```bash
# LangGraph
langgraph dev                 # Local development server
langgraph deploy              # Deploy to LangSmith
langgraph build               # Build deployment artifact

# LangSmith
langsmith login               # Authenticate with LangSmith
langsmith push                # Push dataset to LangSmith
```

---

## Key Links

| Resource | URL |
|---|---|
| LangGraph Docs | https://langchain-ai.github.io/langgraph/ |
| LangChain Docs | https://docs.langchain.com |
| LangSmith Platform | https://smith.langchain.com |
| LangChain Academy (Free) | https://academy.langchain.com |
| LCAE Practice Exam | https://academy.langchain.com/courses/lcae-practice-exam |
| Course: LangChain Foundations | https://academy.langchain.com/courses/foundation-introduction-to-langchain-python |
| Course: Deep Agents Foundations | https://academy.langchain.com/courses/foundation-introduction-to-deepagents |
| Course: Building Reliable Agents | https://academy.langchain.com/courses/building-reliable-agents |
| Course: LangSmith Deployment | https://academy.langchain.com/courses/langsmith-deployment |
| Course: Monitoring Production Agents | https://academy.langchain.com/courses/production-monitoring |
| DeepLearning.AI Course | https://www.deeplearning.ai |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |
| LangChain GitHub | https://github.com/langchain-ai/langchain |
| Talview Exam Pre-Requisites | https://certification.talview.com/support/solutions/articles/11000130864-pre-requisites-for-online-test |
| Exam Support | certification-support@langchain.dev |
| Your GitHub | https://github.com/BigBro2454 |

---

## Progress Tracker

### Phase 1 — Foundations (Weeks 1-3)
- [ ] Week 1: Understand the LangGraph state machine mental model
- [ ] Week 2: State, Nodes, Edges — build your first graph
- [ ] Week 3: Tools and create_agent — first working agent

### Phase 2 — Agent Development (Weeks 4-6)
- [ ] Week 4: Persistence, checkpointers, and threads
- [ ] Week 5: Human-in-the-loop, middleware, and context engineering
- [ ] Week 6: Multi-agent systems and Deep Agents

### Phase 3 — Testing & Evaluation (Weeks 7-8)
- [ ] Week 7: Evaluation fundamentals — datasets, experiments, evaluators
- [ ] Week 8: Online vs. offline evaluation, reliability engineering

### Phase 4 — Deploy & Monitor (Weeks 9-12)
- [ ] Week 9: Deployment fundamentals and LangSmith deployment
- [ ] Week 10: Advanced deployment — multi-region, auth, storage
- [ ] Week 11: Monitoring — traces, online evals, insights, alerts
- [ ] Week 12: Capstone project complete + LCAE practice exam passed

---

*Last updated: September 2026 | LangGraph 0.4+ | LangChain 1.0+ | LCAE Certification Exam Guide*
