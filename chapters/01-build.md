# Section 1: Build (LCAE Exam Study Guide)

## 1. Chapter Overview

Welcome to Section 1 of the LangChain Certified Agent Engineer (LCAE) study guide, focusing on **Build**. This section accounts for 25% of the total exam score (10 questions) and is foundational to your understanding of how to architect, develop, and deploy AI agents using LangChain and LangGraph.

### What This Section Covers
This chapter delves into the core components required to build robust agents:
- The distinction and decision frameworks between standard `create_agent` and the advanced `deepagents` harness.
- Implementing **Middleware** to intercept and modify agent execution flows.
- **Context Engineering** to efficiently manage LLM context windows for long-running processes.
- Utilizing **AGENTS.md and SKILL.md** for declarative agent configuration and capability management.
- Applying **Sandboxing** techniques to securely execute agent-generated code and tool calls.

### The PM Perspective: Why PMs Need to Understand Agent Building
As a Technical Product Manager (TPM), you are not just managing standard web applications; you are overseeing non-deterministic systems. Understanding how agents are built allows you to:
1. **Scope accurately:** Know what is feasible within a single request-response cycle versus what requires a long-running, durable agent.
2. **Define constraints:** Set realistic token budgets, latency expectations, and security perimeters.
3. **Design for failure:** Agents *will* fail. Understanding sandboxing and context management helps you design fallback mechanisms and human-in-the-loop (HITL) workflows.
4. **Bridge the gap:** Translate user requirements into technical architecture (e.g., "We need this agent to remember the last 5 steps" translates to specific context engineering techniques).

### Connection to the Agent Development Lifecycle (ADLC)
The "Build" phase is the core implementation step in the ADLC. It follows Ideation and Design, and precedes Testing, Deployment, and Monitoring. Decisions made here—such as choosing `deepagents` for a durable workflow or implementing a specific middleware for PII redaction—directly impact downstream testing complexity and deployment requirements.

---

## 2. Mental Models for PMs

To effectively manage agent products, use these mental models to bridge technical concepts with product strategy.

### `create_agent` vs `deepagents`: Microservices vs. Monoliths
- **`create_agent` (Microservice):** Fast, lightweight, stateless between sessions, and designed for a specific task. Like a microservice, it handles a quick request, interacts with a few tools, and returns a response. Ideal for chat bots or quick data retrieval.
- **`deepagents` (Monolith/Orchestrator):** Heavy, durable, stateful, and capable of managing complex, multi-hour workflows. It is like an enterprise orchestration platform (e.g., Kubernetes or Airflow) for agents. It handles memory, retries, and deep context natively.

### Middleware: Quality Gates in a CI/CD Pipeline
Think of Middleware as quality gates or interceptors in your CI/CD pipeline. Just as code must pass linting, unit tests, and security scans before deployment, an agent's input and output must pass through middleware for validation, summarization, logging, or redaction before reaching the user or the LLM.

### Context Engineering: Information Architecture for LLMs
In UX design, Information Architecture (IA) dictates how information is structured so users can find what they need without cognitive overload. **Context Engineering** is IA for the LLM. Instead of dumping a 100-page manual into the prompt (cognitive overload), you summarize, prune, and progressively disclose information so the LLM remains focused and cheap to run.

### AGENTS.md & SKILL.md: Product Spec vs. Feature Spec
- **AGENTS.md (Product Spec):** The overarching PRD (Product Requirements Document) for the agent. It defines global rules, persona, and universal constraints.
- **SKILL.md (Feature Spec):** The detailed acceptance criteria and technical spec for a single feature. It defines exactly how an agent should perform a specific sub-task, loaded only when that feature is needed.

### Sandboxing: The Blast Radius
Sandboxing defines the "blast radius" of a compromised or hallucinating agent. If an agent writes a script to delete files, the sandbox ensures it only deletes files within a temporary, isolated container, protecting your core product infrastructure.

---

## 3. First Principles: Requirements & Solution Design

Building an agent starts with rigorous requirements gathering. The non-deterministic nature of LLMs means traditional requirements engineering must be adapted.

### How to Think About Agent Requirements
Start from first principles:
1. **Input Space:** What triggers the agent? (User chat, cron job, webhook?)
2. **Action Space:** What systems can the agent affect? (Read-only APIs, write APIs, executing code?)
3. **Time Horizon:** Does the task take seconds (synchronous) or hours (asynchronous)?
4. **Tolerance for Error:** Is a hallucination mildly annoying (drafting an email) or catastrophic (executing a trade)?

### Decision Framework: When to use `create_agent` vs `deepagents`

| Product Requirement | Recommendation | Rationale |
| :--- | :--- | :--- |
| **Sub-second latency required** | `create_agent` | Lightweight, standard LangGraph execution. |
| **Multi-hour execution times** | `deepagents` | Requires durable execution; standard graphs might timeout or lose state. |
| **Simple Tool Calling (e.g., Weather API)** | `create_agent` | Overhead of `deepagents` is unnecessary for simple retrieval. |
| **Complex, multi-step reasoning with self-correction** | `deepagents` | Native support for advanced context management and long-running state. |
| **Human-in-the-loop approvals required** | Both (Lean `deepagents`) | Both support HITL, but `deepagents` handles pausing for hours better. |

### Solution Design Template for Agent Features

When drafting a solution design, include these specific agent sections:

1. **Agent Persona & Goal:** What is the agent's primary objective?
2. **Architecture Choice:** `create_agent` or `deepagents`? Why?
3. **Tool Registry:** List of tools (APIs, DB access) required.
4. **Middleware Chain:** e.g., PII Redaction (Before) -> Execution -> Summarization (After).
5. **Context Strategy:** How will we prevent token limits from being reached? (e.g., "Use SummarizationMiddleware after 10 turns").
6. **Security & Sandboxing:** How are we isolating code execution?

---

## 4. Deep Dive: `create_agent` vs `deepagents`

Understanding the structural differences between these two agent architectures is critical for the exam and real-world deployment.

### `create_agent` (Standard LangChain/LangGraph)
`create_agent` (often used via `create_react_agent` in `langgraph.prebuilt`) is the factory function for creating standard ReAct (Reasoning and Acting) agents.

**Key Characteristics:**
- **Foundation:** Built directly on top of standard LangGraph `StateGraph`.
- **Execution Pattern:** Short-lived, request-response. Typically runs in a single synchronous or quickly-resolved asynchronous session.
- **State Management:** Uses standard LangGraph checkpointers (e.g., `MemorySaver`, `SqliteSaver`) but requires manual management for very long conversations.
- **Use Case:** Chatbots, simple automation, query answering.

**Example Code:**
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(location: str):
    \"\"\"Get weather for a location.\"\"\"
    return f"Sunny in {location}"

model = ChatOpenAI(model="gpt-4o")
tools = [get_weather]

# Create the standard agent
agent_executor = create_react_agent(model, tools)

# Execute
response = agent_executor.invoke(
    {"messages": [("user", "What is the weather in SF?")]}
)
print(response["messages"][-1].content)
```

### `deepagents` (Advanced Harness)
`deepagents` is an advanced, opinionated harness built *on top of* LangGraph, designed specifically for autonomous, long-running, complex agents that need to operate for minutes to hours.

**Key Characteristics:**
- **Durable Execution:** Built to survive server restarts. It automatically checkpoints aggressive state and can resume precisely where it left off.
- **Automatic Context Management:** Natively implements summarization and pruning to keep token counts manageable over thousands of steps.
- **Declarative Configuration:** Uses `AGENTS.md` and `SKILL.md` to define behavior rather than hardcoding prompts in Python.
- **Progressive Disclosure:** Dynamically loads skills (tools and instructions) only when the agent decides it needs them, saving massive amounts of input tokens.

**Example Code Concept:**
```python
# Conceptual usage of deepagents harness
from deepagents import DeepAgentHarness

harness = DeepAgentHarness(
    workspace_dir="./my_agent_workspace", # Contains AGENTS.md and skills/
    model="claude-3-5-sonnet",
    durability=True
)

task_id = harness.submit_task("Refactor the authentication module and write tests.")
# Returns immediately; agent runs asynchronously in the background.

status = harness.get_status(task_id)
```

### Comparison Summary

| Feature | `create_agent` | `deepagents` |
| :--- | :--- | :--- |
| **Complexity Focus** | Low to Medium | High to Very High |
| **Execution Duration** | Seconds to Minutes | Minutes to Hours/Days |
| **Configuration** | Python code / System Prompts | Declarative (`AGENTS.md`, `SKILL.md`) |
| **Context Management** | Manual / Custom Middlewares | Automatic / Progressive Disclosure |
| **State Resilience** | Standard Checkpointers | High Durability, Crash-Resistant |

**Textual Decision Tree:**
```text
Is the task completed in < 1 minute?
 ├── YES -> Use create_agent
 └── NO -> Does the task require exploring file systems, running code, and self-correcting?
      ├── YES -> Use deepagents
      └── NO -> Use create_agent with asynchronous execution
```

---

## 5. Deep Dive: Middleware

Middleware allows you to intercept and modify the state of an agent at various points in its execution lifecycle. It is essential for production agents to enforce policies, manage costs, and log activity.

### The Execution Lifecycle Hooks
Middleware in LangGraph/LangChain typically intercepts the execution at three key phases:
1. **`before_model` (Pre-processing):** Intercepts the state *before* it is sent to the LLM. 
   - *Use cases:* PII redaction, prompt injection filtering, injecting retrieval context.
2. **`modify_model_request` (Request manipulation):** Modifies the actual API payload going to the LLM provider.
   - *Use cases:* Forcing a specific tool choice, adding provider-specific parameters.
3. **`after_model` (Post-processing):** Intercepts the LLM's response *before* it updates the state or executes a tool.
   - *Use cases:* Response validation, summarization, enforcing JSON schema formatting.

### Built-in Middleware Examples

1. **`SummarizationMiddleware`:** Automatically summarizes older messages in the history to prevent token limits.
2. **`HumanInTheLoopMiddleware`:** Pauses execution if the LLM requests a high-risk tool, waiting for human approval.

### Writing Custom Middleware

Middleware is often implemented as a function that wraps a node in the graph, or as explicit nodes placed before/after the model node.

**Example: PII Redaction Middleware (Conceptual)**
```python
def pii_redaction_middleware(state: dict):
    \"\"\"Intercepts state before sending to LLM.\"\"\"
    messages = state.get("messages", [])
    last_msg = messages[-1]
    
    # Simple regex redaction for SSN
    import re
    redacted_content = re.sub(r'\d{3}-\d{2}-\d{4}', '[REDACTED SSN]', last_msg.content)
    
    # Update state
    last_msg.content = redacted_content
    return {"messages": messages}
```

### Middleware Ordering and Composition
The order of middleware matters significantly.
*Example Pipeline:*
`Decrypt Input -> Redact PII -> Send to LLM -> Log Response -> Validate JSON format`
If you place PII Redaction *after* sending to the LLM, the raw PII has already been sent to the external provider, resulting in a data breach.

### PM Lens: Middleware as Product Guardrails
For a TPM, middleware represents your non-functional requirements (NFRs). When a stakeholder says, "We can't let the agent curse," you translate that to: "We need an `after_model` classification middleware to intercept and rewrite inappropriate responses."

---

## 6. Deep Dive: Context Engineering

Context engineering is the science of managing what information an LLM sees at any given time. Because LLM context windows are finite and costs scale with token counts, you cannot simply feed infinite history into the model.

### Strategies for Context Engineering

1. **Summarization (The Sliding Window with Summary):**
   - Keep the most recent $N$ messages verbatim.
   - Compress all previous messages into a running summary.
   - *Pros:* Maintains long-term narrative without token explosion.
   - *Cons:* Loses fine-grained details from early interactions.

2. **Pruning / Eviction:**
   - Removing specific types of messages from the history (e.g., dropping intermediate tool calls and retaining only the final answer).
   - *Example:* If an agent uses a `search` tool 5 times to find an answer, prune the 5 search queries/results and just keep the final summarized answer in the context.

3. **Progressive Disclosure:**
   - Providing only the metadata or descriptions of tools/files first.
   - The agent must explicitly ask to read the full content if it deems it necessary.
   - *Example:* Instead of loading a 10,000-line codebase, provide a directory tree. The agent uses a tool to read specific files.

4. **Retrieval-Augmented Context (RAG as Context):**
   - Storing past interactions or long documents in a vector database and dynamically injecting relevant snippets into the prompt based on the current user query.

### Token Budget Management
A TPM must define a **Token Budget**. If your context window is 128k tokens, you might partition it:
- 10% (12.8k): System prompt and global instructions.
- 20% (25.6k): Active tool descriptions (SKILL.md).
- 50% (64k): Working memory / Document context.
- 20% (25.6k): Buffer for LLM output.

### PM Lens: UX Optimization for the LLM
Just as a cluttered UI confuses a user, a cluttered prompt confuses an LLM (causing hallucinations or forgotten instructions). Context engineering ensures the agent has exactly what it needs to succeed—no more, no less.

---

## 7. Deep Dive: AGENTS.md and SKILL.md

In `deepagents` and complex LangGraph implementations, managing prompts as Python strings becomes untenable. The industry standard is moving towards declarative markdown files.

### AGENTS.md (Global Rules)
`AGENTS.md` sits at the root of your project. It acts as the global system prompt and operational manual.

**Structure & Content:**
- **Persona:** Who the agent is.
- **Global Rules:** Constraints that apply universally (e.g., "Never modify code outside the /src directory").
- **Workflow:** High-level steps the agent should always follow.

**Example `AGENTS.md`:**
```markdown
# Core Architecture Agent

You are a senior software architect responsible for refactoring the legacy python codebase.

## Global Rules
1. ALWAYS write unit tests for any new function you create.
2. NEVER delete existing comments unless they are explicitly wrong.
3. Use the `git_commit` tool after every logical milestone.

## Operational Workflow
1. Analyze the requested change.
2. Search the codebase for affected components.
3. Propose a design document (write to `design.md`).
4. Execute the refactoring.
```

### SKILL.md (Progressive Disclosure Capabilities)
A "Skill" represents a specific capability (like "Database Migration" or "API Integration"). Instead of loading all skills into the prompt at once, `deepagents` uses **Progressive Disclosure**.

1. **Phase 1 (Discovery):** The agent sees only the YAML frontmatter of all available skills.
2. **Phase 2 (Loading):** If the agent needs to migrate a database, it calls a tool to "load skill", which injects the full `SKILL.md` content into its context.

**Directory Structure:**
```text
workspace/
├── AGENTS.md
└── skills/
    ├── database_migration/
    │   ├── SKILL.md
    │   └── scripts/
    │       └── run_migration.sh
    └── frontend_testing/
        ├── SKILL.md
        └── templates/
```

**Example `SKILL.md`:**
```markdown
---
name: Database Migration
description: Use this skill when you need to alter database tables or create new schemas.
tools: [run_sql, generate_migration]
---

# Database Migration Guide

When performing migrations on this project, you must follow these specific steps:
1. Always generate a rollback script.
2. Use the `run_sql` tool with the `--dry-run` flag first.
3. Ensure no data-loss operations are performed without user confirmation.

## Allowed Data Types
- UUID (for primary keys)
- VARCHAR(255)
- TIMESTAMP WITH TIME ZONE
```

### Why this matters
By keeping detailed instructions out of the main prompt until necessary, you save thousands of tokens per request and prevent the LLM from getting confused by irrelevant instructions.

---

## 8. Deep Dive: Sandboxing

When agents are given tools that can interact with the outside world (executing code, writing files, making network requests), **Sandboxing** is critical.

### Why Sandboxing Matters
LLMs hallucinate. A bad prompt or a malicious user input (Prompt Injection) could cause an agent to execute `rm -rf /` or exfiltrate sensitive environment variables.

### Types of Sandboxing
1. **Code Execution Sandbox:** 
   - Running generated Python/Bash code in an isolated environment (e.g., Docker containers, WebAssembly, or specialized platforms like E2B).
   - Prevents the agent from accessing the host machine's OS.
2. **File System Access:**
   - Chroot jails or restricted volume mounts.
   - Ensures the agent can only read/write files in a specific `workspace/` directory.
3. **Network Access:**
   - Restricting outbound API calls via firewalls or proxies.
   - *Example:* The agent can query `api.github.com` but cannot make requests to unknown IP addresses.

### LangChain/LangGraph Approach
LangGraph does not natively run Docker containers; it delegates this to tools. You must build or integrate sandboxed tools. For instance, using the `langchain-experimental` Python REPL tool requires caution, and in production, should be replaced with a remote execution API (like E2B or a custom AWS Lambda).

### Threat Model for Agent Products
As a PM, evaluate:
- **Confidentiality:** Can the agent read secrets? (Ensure env vars are not exposed in the sandbox).
- **Integrity:** Can the agent corrupt the main database? (Give the agent a sandbox database or read-only credentials).
- **Availability:** Can the agent spin in an infinite loop consuming compute? (Set strict iteration limits in LangGraph).

---

## 9. Hands-On Projects

To prepare for the LCAE exam, complete these conceptual projects.

### Project B.1: Build a `create_agent`
**Objective:** Create a basic ReAct agent with 3 tools.
**Instructions:**
1. Define three tools: `get_time`, `calculate_math`, `search_wiki`.
2. Use `create_react_agent` from `langgraph.prebuilt`.
3. Invoke the agent with a complex query requiring at least 2 tools.
**What to Learn:** Understand the state structure and how tool calls are appended to the `messages` list.

### Project B.2: Implement Middleware
**Objective:** Add a logging middleware that tracks tool usage.
**Instructions:**
1. Create a function that intercepts the state before the model node.
2. If the last message is a `ToolMessage`, log the tool name and output length to a file.
**What to Learn:** Master the LangGraph node execution order and state manipulation.

### Project B.3: AGENTS.md and SKILL.md Configuration
**Objective:** Write configuration files for a "Data Analyst" agent.
**Instructions:**
1. Write `AGENTS.md` defining the persona (Data Analyst) and global rules (always use pandas).
2. Write a `skills/clean_data/SKILL.md` with YAML frontmatter and specific steps for handling null values.
**What to Learn:** Practice declarative prompt engineering and progressive disclosure concepts.

### Project B.4: Basic Sandboxing Strategy
**Objective:** Design a secure code execution environment.
**Instructions:**
1. Write a Python tool that accepts code from the LLM.
2. Instead of using `exec()`, write the code to a `.py` file and execute it inside a newly spawned, network-restricted Docker container.
**What to Learn:** Understand the overhead and latency implications of secure sandboxing versus native execution.

---

## 10. Exam Prep: Key Takeaways

### Summary Table

| Concept | Exam Focus | Key Takeaway |
| :--- | :--- | :--- |
| **`create_agent`** | When to use it | Best for simple, short-lived, synchronous tasks. |
| **`deepagents`** | When to use it | Best for long-running, durable, complex orchestration. |
| **Middleware** | Execution hooks | Modifies state `before_model`, `modify_model_request`, `after_model`. |
| **Context Eng.** | Token limits | Use summarization and pruning to keep context windows efficient. |
| **AGENTS/SKILL** | Declarative config | `AGENTS.md` = global rules; `SKILL.md` = progressive disclosure. |
| **Sandboxing** | Security | Never run agent-generated code natively; use Docker/Wasm isolation. |

### Common Exam Question Patterns
- **Scenario-based Architecting:** "You are building an agent that needs to analyze a 10GB log file over 4 hours. Which architecture do you choose?" (Answer: `deepagents` with sandboxed file access).
- **Middleware Ordering:** "Where should a tool-validation middleware be placed?" (Answer: `after_model` to intercept the LLM's requested tool call before execution).
- **Context Optimization:** "Your agent is failing due to token limits after 50 turns. What is the best solution?" (Answer: Implement a summarization middleware or state pruning strategy).

### Gotchas and Edge Cases
- **Tool Output size:** A common failure point is a tool returning a massive payload (e.g., `SELECT * FROM users`). Ensure middleware truncates or summarizes large tool outputs before they hit the LLM context.
- **Progressive Disclosure:** Remember that the LLM only sees the YAML metadata of a `SKILL.md` initially. If the description is poor, the agent will never load the skill.

---

## 11. Resources

- [LangChain Foundations Course](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python)
- [Deep Agents Foundations Course](https://academy.langchain.com/courses/foundation-introduction-to-deepagents)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Security Best Practices](https://python.langchain.com/docs/security)

*Good luck on the Build section of the LCAE exam! Remember to approach the questions by balancing technical feasibility with product requirements.*
