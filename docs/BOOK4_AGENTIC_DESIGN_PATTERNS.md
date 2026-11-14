# Agentic Design Patterns Handbook — Reference & Implementation Guide

> **Author**: Felix Williams  
> **Source Reference**: *Agentic Design Patterns Handbook: The Ultimate Guide to Building Autonomous AI Agents* (Book 4)  
> **Framework**: LangGraph 0.2+ / LangChain 0.3+  
> **Implementation Directory**: [`langgraph/agents/`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents)

---

## Executive Summary

In 2025–2026, autonomous AI agents have moved beyond research prototypes into mission-critical systems moving money, writing code, resolving customer issues, and executing long-horizon analytical workflows. The critical differentiator between agents powering billion-dollar platforms and those failing in unpredictable loops is **disciplined architecture**.

This guide synthesizes the architectural patterns, state machines, decision matrices, benchmark lifts, and safety protocols from Felix Williams' field manual, accompanied by runnable **LangGraph** implementations.

```
                   ┌──────────────────────────────────────────────────────────┐
                   │             The Production Agent Stack                   │
                   ├──────────────────────────────────────────────────────────┤
                   │  Layer 4: Multi-Agent Orchestration                      │
                   │  - Supervisor-Crew    - Hierarchical Teams               │
                   │  - Debate/Adversarial - Market-Based / Contract Net      │
                   │  - Sequential Pipeline- Parallel Execution (Fan-Out/In)  │
                   ├──────────────────────────────────────────────────────────┤
                   │  Layer 3: Single-Agent Reasoning & Search                │
                   │  - ReAct (Reason+Act) - Plan-and-Execute                 │
                   │  - Reflexion (Verbal) - Tree-of-Thought (ToT)            │
                   │  - Monte Carlo Tree Search (MCTS)                        │
                   ├──────────────────────────────────────────────────────────┤
                   │  Layer 2: Reliability, Guardrails & Memory               │
                   │  - 3-Layer Guardrails (Input / Reasoning / Output)       │
                   │  - Constitutional AI Chains & Auto-Remediation           │
                   │  - Human-in-the-Loop (HITL) Interrupts & Rollbacks       │
                   │  - Episodic Reflection Buffers & Checkpointers           │
                   ├──────────────────────────────────────────────────────────┤
                   │  Layer 1: Foundational Execution                         │
                   │  - StateGraph State Machines, Reducers & Tool Nodes      │
                   │  - Multi-Provider Resilience (Cloud LLM + Local Ollama)  │
                   └──────────────────────────────────────────────────────────┘
```

---

## Chapter 1: Anatomy of an Autonomous Agent

An autonomous agent is a software system that **perceives** its environment, **reasons** about objectives, takes **actions** to fulfill them, and **learns** from outcomes with minimal or no human intervention for routine operations.

### 1.1 Four Fundamental Properties
1. **Autonomy**: Operates independently without requiring turn-by-turn human supervision.
2. **Reactivity**: Perceives environment state changes, tool outputs, and exceptions, responding dynamically.
3. **Proactivity**: Takes initiative toward achieving high-level goals rather than merely echoing inputs.
4. **Social Ability**: Interacts effectively with humans (HITL gates) and other peer agents via typed state contracts.

### 1.2 Evolution: Assistant vs. Copilot vs. Autonomous Agent

| Capability Dimension | Traditional Chatbot | AI Copilot | Autonomous Agent |
| :--- | :--- | :--- | :--- |
| **User Interaction** | Turn-by-turn text | Suggests inline completions | Accepts high-level goal, executes loop |
| **Tool Execution** | None (pure text generation) | Single-turn tools via human click | Autonomous dynamic tool chaining |
| **Planning Horizon** | Zero (stateless) | 1-step immediate context | Multi-step strategic planning & replanning |
| **Memory Architecture**| Ephemeral prompt buffer | File / Workspace context | Working memory + Episodic reflections |
| **Error Recovery** | Fails silently / hallucinates | Relies on human to spot error | Self-corrects, retries, evaluates, escalates |
| **Latency Profile** | Sub-second | Milliseconds to seconds | Seconds to hours |

---

## Chapter 2: Single-Agent Reasoning Patterns

```
                                    SINGLE-AGENT DECISION TREE
                                                │
                                  Is task linear and <8 steps?
                                    ├── YES ──► Use ReAct
                                    └── NO
                                         │
                         Does task require >5 coordinated steps?
                                    ├── YES ──► Use Plan-and-Execute
                                    └── NO
                                         │
                         Is there a reliable pass/fail test/evaluator?
                                    ├── YES ──► Use Reflexion
                                    └── NO
                                         │
                         Does problem require exploring multiple branches?
                                    ├── YES ──► Use Tree-of-Thought (ToT)
                                    └── NO
                                         │
                         Is it high-stakes code/game with large search space?
                                    └── YES ──► Use Monte Carlo Tree Search (MCTS)
```

---

### 2.1 ReAct (Reason + Act)

- **Source Reference**: Book 4 Chapter 2.1
- **Code Implementation**: [`agents/react_pattern/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/react_pattern/agent.py)
- **Core Loop**: `START -> Reason (Thought) -> Tools (Action) -> Reason (Observation) -> ... -> END`

#### Key Characteristics
- Interleaves reasoning traces with tool actions in a single prompt loop.
- The agent formulates a thought explaining *why* a tool is needed, executes the tool, inspects the observation, and decides whether to continue or emit the final answer.
- **Circuit Breaker**: Turn limits (e.g. `MAX_TURNS = 6`) protect against infinite loops.

```
                    ┌───────────────────────────────┐
                    │             START             │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
             ┌─────►│          Reason Node          │◄─────┐
             │      │   - Generate Thought          │      │
             │      │   - Call Tool or Answer       │      │
             │      └──────────────┬────────────────┘      │
             │                     │                       │
      Tool calls present?          │                       │
             │                     ▼                       │
             │      ┌───────────────────────────────┐      │
             ├──────┤        tools_condition        │      │
             │      └──────────────┬────────────────┘      │
             │                     │                       │
             │              No tool calls                  │
             │                     │                       │
             │                     ▼                       │
             │      ┌───────────────────────────────┐      │
             │      │              END              │      │
             │      └───────────────────────────────┘      │
             │                                             │
             └──────────────── ToolNode ───────────────────┘
```

#### Run CLI Command
```bash
python runners/run_agent.py react "Calculate (45 * 12) + (180 / 4)"
```

---

### 2.2 Plan-and-Execute

- **Source Reference**: Book 4 Chapter 2.2
- **Code Implementation**: [`agents/plan_and_execute/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/plan_and_execute/agent.py)
- **Core Loop**: `Planner -> Executor (Step N) -> Replanner (Evaluate & Adapt) -> Executor (Step N+1) ... -> Final Response`

#### Architectural Trade-offs
- **Decoupled Architecture**: Separates strategic, high-level planning from tactical, low-level execution.
- **Adaptive Replanning**: After each subtask completes, the replanner inspects intermediate outputs. If new data invalidates the initial plan, it replans dynamically without losing completed progress.
- **Benchmark Lift**: +25% to +40% task completion lift over basic ReAct on multi-step workflows.

#### Run CLI Command
```bash
python runners/run_agent.py plan_and_execute "Design a zero-downtime database migration from MongoDB to PostgreSQL"
```

---

### 2.3 Reflexion (Verbal Self-Reflection)

- **Source Reference**: Book 4 Chapter 2.3
- **Code Implementation**: [`agents/reflexion/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/reflexion/agent.py)
- **Core Loop**: `Actor (Trial N) -> Evaluator -> [Failed] -> Self-Reflector (Extract Lesson) -> Episodic Buffer -> Actor (Trial N+1)`

#### Architectural Trade-offs
- **Reinforcement via Verbal Memory**: Replaces costly parameter updates with natural language self-reflections conditioned into subsequent prompt trials.
- **Benchmark Lift**: Improves HumanEval pass rates from 67% to 91% (+36% relative lift); SWE-Bench resolved rate 30–38%.
- **Stopping Criteria**: Passes evaluation tests OR exhausts retry budget (e.g. `MAX_TRIALS = 3`).

#### Run CLI Command
```bash
python runners/run_agent.py reflexion "Write a Python function to find the median of two sorted arrays in O(log(m+n))"
```

---

### 2.4 Tree-of-Thought (ToT)

- **Source Reference**: Book 4 Chapter 2.4
- **Code Implementation**: [`agents/tree_of_thought/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/tree_of_thought/agent.py)
- **Core Loop**: `Branch (Generate K candidates) -> Evaluate (Score 1-10) -> Prune (Beam Width B) -> Depth Loop -> Synthesize Solution`

#### Architectural Trade-offs
- Models reasoning as explicit tree search rather than single linear chain.
- Proposer expands $K$ candidate thoughts; Evaluator scores each node; Pruner keeps top $B$ paths and discards unviable branches.
- **Benchmark Lift**: 74% vs. 4% solve rate on Game of 24 mathematical reasoning; 65–75% human preference lift on multi-constraint planning.

#### Run CLI Command
```bash
python runners/run_agent.py tree_of_thought "Architect a multi-region disaster recovery strategy for payments"
```

---

### 2.5 Monte Carlo Tree Search (MCTS)

- **Source Reference**: Book 4 Chapter 2.5
- **Code Implementation**: [`agents/mcts/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/mcts/agent.py)
- **Core Loop**: 
  1. **Selection**: Traverse tree using UCB1 formula $Q + c \sqrt{\frac{\ln N}{n}}$.
  2. **Expansion**: Generate new child reasoning steps.
  3. **Simulation / Rollout**: Fast mental rollout assessing terminal reward (0.0 to 1.0).
  4. **Backpropagation**: Increment visit count $N$ and cumulative value $Q$ along parent trajectory.
  5. **Extraction**: Greedily extract path with highest visit count.

#### Benchmark Lift
- 42%–49% solve rate on SWE-Bench Verified; 2× solve rate on competitive reasoning benchmarks.

#### Run CLI Command
```bash
python runners/run_agent.py mcts "Architect an ultra-low latency inference cache for streaming LLM tokens"
```

---

## Chapter 3: Multi-Agent Orchestration Patterns

```
                                  MULTI-AGENT PATTERN SELECTION
                                                │
                                  What is the primary constraint?
                                                │
         ┌──────────────────┬───────────────────┼───────────────────┬──────────────────┐
         ▼                  ▼                   ▼                   ▼                  ▼
    Routing Domain    10+ Sub-Teams        High-Stakes          Cost / Latency    Fixed Pipeline
    Specialists         Hierarchy          Verification         Optimization       ETL / Docs
         │                  │                   │                   │                  │
         ▼                  ▼                   ▼                   ▼                  ▼
    Supervisor-Crew    Hierarchical        Debate /            Market-Based /      Sequential
    Hub-and-Spoke         Teams           Adversarial          Contract Net         Pipeline
```

---

### 3.1 Supervisor-Crew Pattern

- **Source Reference**: Book 4 Chapter 5.1
- **Code Implementation**: [`agents/supervisor/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/supervisor/agent.py)
- **Topology**: Hub-and-Spoke. Central supervisor routes subtasks to specialized domain workers (e.g. `Researcher`, `Writer`), receives structured outputs, and decides when the objective is satisfied.
- **Best For**: 2–5 worker agents, domain routing, content and briefing generation.

---

### 3.2 Hierarchical Teams

- **Source Reference**: Book 4 Chapter 5.2
- **Code Implementation**: [`agents/hierarchical_teams/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/hierarchical_teams/agent.py)
- **Topology**: Multi-level organizational tree.
  - **Level 0**: Executive Director / Top-Level Orchestrator
  - **Level 1**: Research Team Lead & Engineering Systems Lead
  - **Level 2**: Specialist Workers (Domain Data Collector, Systems Architect, Compliance Verifier)
- **Best For**: Complex cross-functional enterprise initiatives (10+ agents). Natural domain isolation prevents context overload.

#### Run CLI Command
```bash
python runners/run_agent.py hierarchical_teams "Design an autonomous anti-money laundering (AML) detection platform"
```

---

### 3.3 Debate / Adversarial Pattern

- **Source Reference**: Book 4 Chapter 5.3
- **Code Implementation**: [`agents/debate_adversarial/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/debate_adversarial/agent.py)
- **Topology**: Dialectical Triad.
  - **Proponent**: Constructs the affirmative proposal and defends against critique.
  - **Opponent Critic**: Uncompromisingly attacks logical gaps, vulnerabilities, and edge cases.
  - **Arbiter / Judge**: Impartial arbitrator evaluating the multi-round transcript and delivering a consensus ruling.
- **Benchmark Lift**: Reduces hallucination rates by **30% to 50%**.

#### Run CLI Command
```bash
python runners/run_agent.py debate "Should mission-critical financial workflows use monolithic or multi-agent architectures?"
```

---

### 3.4 Market-Based Pattern (Contract Net Protocol)

- **Source Reference**: Book 4 Chapter 5.4
- **Code Implementation**: [`agents/market_based/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/market_based/agent.py)
- **Mechanism**:
  1. **Task Broker** publishes Request For Proposal (RFP) with optimization priority (`cost`, `speed`, `quality`, or `balanced`).
  2. **Heterogeneous Bidders** submit structured bids with confidence, estimated token cost, latency, and methodology.
  3. **Auction Clearing Engine** computes utility scores and awards contract to best-fit agent.
  4. **Contractor Execution & Settlement** generates deliverable and transaction audit receipt.
- **Cost Reduction**: Dynamically reduces system-wide token spend by **40% to 60%**.

#### Run CLI Command
```bash
python runners/run_agent.py market_based "Perform high-rigor vulnerability scanning and compliance audit for healthcare data API"
```

---

### 3.5 Sequential Pipeline

- **Source Reference**: Book 4 Chapter 5.5
- **Code Implementation**: [`agents/sequential_pipeline/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/sequential_pipeline/agent.py)
- **Topology**: Assembly Line:
  `Stage 1 (Ingest & Cleanse) -> Stage 2 (Entity Extraction) -> Stage 3 (Risk Audit) -> Stage 4 (Deliverable Packaging)`
- **Features**: Strongly typed state schemas with intermediate artifact checkpointing via `MemorySaver`.

#### Run CLI Command
```bash
python runners/run_agent.py sequential_pipeline "Customer Incident: Payment gateway timeout during peak Black Friday checkout"
```

---

### 3.6 Parallel Execution (Fan-Out / Fan-In)

- **Source Reference**: Book 4 Chapter 5.6
- **Code Implementation**: [`agents/parallel_execution/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/parallel_execution/agent.py)
- **Topology**: Concurrent Scatter-Gather.
  - `START` triggers 3 parallel auditors simultaneously:
    1. `SecurityAuditor`
    2. `PerformanceAuditor`
    3. `CostFinOpsAuditor`
  - All 3 converge at `fan_in_synthesizer` to produce a unified CTO scorecard.
- **Latency Advantage**: Execution time bounded by $O(\max(\text{latency}))$ instead of serial $O(\sum(\text{latency}))$.

#### Run CLI Command
```bash
python runners/run_agent.py parallel_execution "Migrate core auth service from monolith to distributed microservices"
```

---

## Chapter 4: Reliability, Safety & Guardrail Patterns

```
                          3-LAYER DEFENSIVE GUARDRAIL PIPELINE
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   Layer 1: Input Defense Guardrail    │
                        │   - Prompt Injection Detection        │
                        │   - Scope & Boundary Filtering        │
                        │   - Sanitization & Heuristic Scans    │
                        └───────────────────┬───────────────────┘
                                            │
                                  Passed Layer 1?
                                   ├── NO ──► REJECT IMMEDIATELY
                                   └── YES
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │  Layer 2: Reasoning & Execution Guard │
                        │  - Tool Parameter Schema Validation   │
                        │  - Turn & Budget Circuit Breakers     │
                        │  - Sandboxed Tool Invocation          │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   Layer 3: Constitutional AI Auditor  │
                        │   - Factuality & Grounding Check      │
                        │   - PII & System Leak Screening       │
                        │   - Toxic / Hostile Tone Filter       │
                        └───────────────────┬───────────────────┘
                                            │
                               Adheres to Constitution?
                                   ├── NO ──► REMEDIATE & REWRITE
                                   └── YES
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │       Verified Final Delivery         │
                        └───────────────────────────────────────┘
```

- **Source Reference**: Book 4 Chapters 7.3, 8.1, 8.2, 8.3
- **Code Implementation**: [`agents/guardrails_constitutional/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/guardrails_constitutional/agent.py)
- **HITL Integration**: [`agents/hitl_agent/agent.py`](file:///Users/ishan03/Workspace/projects/ai/langgraph/agents/hitl_agent/agent.py) using native LangGraph `interrupt()` and `Command(resume=...)`.

---

## Pattern Comparison Matrix

| Pattern Name | Book 4 Chapter | Agent Type | Primary Use Case | SWE-Bench / GAIA Lift | Relative Cost | Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ReAct** | Ch 2.1 | Single-Agent | Fast tool use, calculations, facts | Baseline | $1.0\times$ | Low |
| **Plan-and-Execute** | Ch 2.2 | Single-Agent | 5–20 step workflows, migrations | $+25\text{--}40\%$ completion | $1.3\times$ | Medium |
| **Reflexion** | Ch 2.3 | Single-Agent | Code generation, verifiable math | $+36\%$ rel. (HumanEval 91%) | $2.0\text{--}3.0\times$ | Med-High |
| **Tree-of-Thought** | Ch 2.4 | Single-Agent | Constraint problems, DR strategy | $74\%$ vs $4\%$ (Game of 24) | $3.0\text{--}5.0\times$ | High |
| **MCTS** | Ch 2.5 | Single-Agent | High-stakes code, complex searches| $42\text{--}49\%$ SWE-Bench | $5.0\text{--}15.0\times$| Very High |
| **Supervisor-Crew**| Ch 5.1 | Multi-Agent | Domain routing, 2–5 specialists | Task-dependent | $1.5\text{--}2.5\times$ | Medium |
| **Hierarchical Teams**| Ch 5.2| Multi-Agent | Cross-functional org workflows | Scales linearly | $3.0\text{--}6.0\times$ | High |
| **Debate / Adversarial**| Ch 5.3| Multi-Agent | Compliance, security architecture | $-30\text{--}50\%$ hallucination | $2.0\text{--}4.0\times$ | Medium |
| **Market-Based** | Ch 5.4 | Multi-Agent | Variable workloads, multi-model | $-40\text{--}60\%$ token cost | Variable | High |
| **Sequential Pipeline**| Ch 5.5| Multi-Agent | Data processing, ETL, tickets | Deterministic throughput | $1.2\text{--}1.8\times$ | Low-Med |
| **Parallel Execution** | Ch 5.6| Multi-Agent | Latency-critical multi-audits | $O(\max(\text{latency}))$ | $N\times$ | Medium |
| **3-Layer Guardrails** | Ch 8.1| Safety Stack| Injection defense, compliance | $<0.1\%$ policy violation | $+15\text{--}20\%$ | Medium |
| **Human-in-the-Loop** | Ch 8.3| Safety Stack| Financial/Production approvals | $100\%$ oversight on gates | Human latency | Medium |

---

## Verification & Test Suite

Run the full automated test suite to verify graph compilation, edge topology, and state transitions across all implemented patterns:

```bash
# Verify environment and agent registry
python main.py

# Run automated tests
pytest evals/test_book4_patterns.py -v
```
