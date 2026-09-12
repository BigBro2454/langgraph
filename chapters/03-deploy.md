# Section 3: Deploy (LCAE Study Guide)

## 1. Chapter Overview
This section covers the crucial transition from developing an agent locally to running it in production using LangSmith and LangGraph Cloud. For a Technical Product Manager, deployment is where your agent transforms from a proof-of-concept into a tangible product that delivers value to users. We will explore deployment architectures, configuration management, security (secrets and auth), and how state is durably persisted.

**The Gap Between 'Works on My Laptop' and 'Works in Production'**
When building locally, developers often run agents synchronously, using local memory for checkpoints, hardcoding API keys, and ignoring concurrent user access. In production, an agent must:
- Handle concurrent requests across many users without state bleeding.
- Ensure high availability and fault tolerance.
- Securely manage API keys and credentials.
- Persist conversation histories durably so a server restart doesn't wipe user data.
- Enforce strict authentication and authorization boundaries.

This chapter bridges that gap.

## 2. Mental Models for PMs
To make technical deployment concepts accessible, let's map them to familiar PM concepts:

- **Graph** = Your product's source code / blueprint. It defines the "what" and "how".
- **Deployment** = Your production environment / live service. The running factory built from your blueprint.
- **Revision** = A version release / deploy (like a git tag going live). A specific iteration of your product.
- **Assistant** = A product variant / SKU (same code, different config). E.g., a "Pro" version vs. "Basic" version, running the same codebase but with different system prompts or models.
- **Thread** = A user session / conversation. The ongoing relationship and context with a specific user.
- **Run** = A single API request / transaction. One turn of the conversation or one execution of a task.

**Infrastructure Analogies:**
- **Multi-region** = Opening offices in different countries. You want data and processing to be close to the customer for speed, and to adhere to local laws (data residency).
- **Secrets** = The keys to your building. You don't leave them out in the open; you lock them in a safe and only give them to trusted personnel (the running service).
- **AuthN vs AuthZ** = The bouncer at the door (AuthN: "Are you on the list?") vs. the VIP list inside (AuthZ: "You're in, but can you access the VIP lounge?").
- **Storage durability** = Data doesn't disappear when the lights go out. It's the guarantee that your users won't lose their work if a server crashes.

## 3. First Principles: Requirements & Solution Design
As a PM, you are responsible for defining the non-functional requirements (NFRs) that dictate the deployment architecture.

### Solution Design Template
When planning an agent deployment, follow this framework:
1. **Target Audience & Traffic:** B2B vs B2C. Expected DAU/MAU. Peak transactions per second (TPS).
2. **Data Sensitivity:** Does the agent process PII, PHI, or strict corporate IP?
3. **Latency Expectations:** Is it a chat interface (needs <2s time-to-first-token) or a background summarization task (minutes are acceptable)?
4. **Residency Requirements:** Must the data stay in the US, EU, or on-premises?

### Deployment Readiness Checklist
- [ ] **Code:** `langgraph.json` configured correctly. Graph compiles without errors.
- [ ] **Infrastructure:** Selected Cloud, Hybrid, or Self-hosted.
- [ ] **Secrets:** All API keys moved to secure environment variables. No hardcoded credentials.
- [ ] **Data:** Checkpoint sizes analyzed to ensure they fall within limits (<25MB per payload).
- [ ] **Auth:** Authentication gateway established. Authorization checks (RBAC) implemented for multi-tenant threads.
- [ ] **Ops:** Alerting and monitoring set up in LangSmith.

### Capacity Planning Worksheet
Agents are compute-heavy and I/O bound (waiting on LLMs).
- **Tokens/Run:** Average tokens per interaction.
- **Cost/Run:** Calculated based on LLM pricing.
- **Concurrent Threads:** How many active users communicating simultaneously?
- **Storage Growth:** Checkpoints per thread * Size per checkpoint * Active threads.

## 4. Deep Dive: Deployment Terminology (THE MOST IMPORTANT SECTION)

This section covers the core taxonomy you MUST memorize for the exam.

### 1. Graph
- **What it is:** The compiled LangGraph agent. It is the definition of nodes, edges, state schema, and tools.
- **How it's defined:** A Python file containing the `StateGraph` compilation, plus a `langgraph.json` configuration file that points to it.
- **PM Perspective:** This is your core intellectual property. The graph dictates the logic and capabilities of the product.
- **Common Mistakes:** Forgetting to expose the compiled graph object or pointing the `langgraph.json` to the wrong variable.
- **Example code (`agent.py`):**
  ```python
  from langgraph.graph import StateGraph, START, END
  from typing import TypedDict
  
  class State(TypedDict):
      messages: list
      
  def my_node(state: State):
      return {"messages": ["Hello!"]}
      
  graph_builder = StateGraph(State)
  graph_builder.add_node("my_node", my_node)
  graph_builder.add_edge(START, "my_node")
  graph_builder.add_edge("my_node", END)
  
  # This compiled graph is what gets deployed
  app = graph_builder.compile()
  ```

### 2. Deployment
- **What it is:** A running instance of a graph on LangSmith infrastructure.
- **Lifecycle:** Created -> Active -> Updated (new revisions) -> Decommissioned.
- **PM Perspective:** The live environment. When a deployment is down, the product is down.
- **Common Mistakes:** Treating a deployment as static. Deployments will have multiple revisions over their lifecycle.

### 3. Revision
- **What it is:** An immutable, versioned update to a deployment.
- **How it works:** When you push new code or change environment variables, a new revision is created.
- **PM Perspective:** The safety net. If a new model update breaks the agent, you can roll back to the previous revision instantly.

### 4. Assistant
- **What it is:** A configured instance of a deployed graph.
- **Use Case:** One codebase, multiple product variants.
- **PM Perspective:** This allows you to monetize different tiers (e.g., Free users get an Assistant configured with Claude 3 Haiku, Pro users get an Assistant configured with Claude 3.5 Sonnet) without rewriting code.
- **Example:**
  ```python
  from langgraph_sdk import get_client
  client = get_client()
  
  # Create an assistant from a deployed graph
  assistant = await client.assistants.create(
      graph_id="my-support-graph",
      config={"configurable": {"system_prompt": "You are a helpful assistant.", "model": "gpt-4o"}}
  )
  ```

### 5. Thread
- **What it is:** A persistent conversation or session, identified by a `thread_id`.
- **PM Perspective:** The user's workspace. A user expects their thread to remember past context.
- **Common Mistakes:** Sharing threads across users (severe security violation) or letting threads grow unbounded without summarizing.
- **Example:**
  ```python
  # Create a thread
  thread = await client.threads.create()
  print(thread["thread_id"])
  ```

### 6. Run
- **What it is:** A single execution/invocation of the graph within a thread.
- **States:** Pending -> Running -> Completed (or Failed/Timeout).
- **PM Perspective:** The unit of work and often the unit of cost.

### FULL End-to-End Workflow Code
```python
import asyncio
from langgraph_sdk import get_client

async def main():
    client = get_client()
    
    # 1. We assume the Deployment and Revision are already active via CLI (`langgraph deploy`)
    # 2. Create an Assistant
    assistant = await client.assistants.create(
        graph_id="agent", 
        config={"configurable": {"model": "gpt-4o"}}
    )
    
    # 3. Create a Thread
    thread = await client.threads.create()
    
    # 4. Create and stream a Run
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant["assistant_id"],
        input={"messages": [{"role": "user", "content": "Hello, what can you do?"}]}
    ):
        if chunk.event == "events":
            print(chunk.data)

if __name__ == "__main__":
    asyncio.run(main())
```

### CRITICAL: Entity Relationship Diagram
```text
Graph (1) --has--> (many) Deployments
Deployment (1) --has--> (many) Revisions
Deployment (1) --has--> (many) Assistants
Assistant (1) --has--> (many) Threads
Thread (1) --has--> (many) Runs
```

## 5. Deep Dive: Multi-Region Serving

Deployment locations matter for speed (latency) and legal compliance (data residency).

| Feature | Cloud (Managed) | Hybrid (Bring Your Own Cloud) | Self-Hosted (On-Prem / Air-gapped) |
| :--- | :--- | :--- | :--- |
| **Control Plane** | LangSmith manages | LangSmith manages | You manage |
| **Data Plane (Execution)**| LangSmith VPC | Your VPC (AWS/GCP/Azure) | Your Datacenter / VPC |
| **Data Residency** | LangSmith regions | Stays in your VPC | Stays on your hardware |
| **Ops Burden** | Minimal | Medium | High |
| **Best For** | Startups, standard SaaS | Enterprise, Healthcare (HIPAA) | Gov, Defense, strict Finance |

### Decision Tree
- **Is the environment completely disconnected from the internet?** -> Self-Hosted.
- **Must data strictly reside in your cloud account (e.g., HIPAA)?** -> Hybrid.
- **Is it a standard web app without strict data isolation?** -> Cloud.

### Real-world Scenarios & Compliance
- **SOC2:** Easily achievable on Cloud if you accept LangSmith's subprocessor terms, but Hybrid gives you more direct control over audit logs and data access.
- **GDPR:** Use an EU region Cloud deployment, or Hybrid within an EU VPC.
- **HIPAA:** Hybrid is highly recommended so PHI never leaves your infrastructure.

## 6. Deep Dive: Secret Handling

Secrets (API keys, DB passwords, OAuth tokens) must NEVER be hardcoded.

### Best Practices & Advanced Strategies
1. **Never hardcode.**
2. **Environment Variables:** Inject securely via deployment config.
3. **Rotation:** Set up automated rotation for DB passwords and API keys. LangSmith allows updating secrets via new revisions.
4. **Audit Logging:** Monitor who accesses secrets and when they are changed.
5. **Separation of Concerns:** Maintain separate `langgraph.json` or `.env` files for `dev`, `staging`, and `prod`.

**Configuration Example (langgraph.json):**
```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:app"
  },
  "env": ".env.production"
}
```

## 7. Deep Dive: Authentication vs Authorization

| Concept | Authentication (AuthN) | Authorization (AuthZ) |
| :--- | :--- | :--- |
| **Question** | "Who are you?" | "What can you do?" |
| **Analogy** | Checking ID at the door | Checking VIP pass for the lounge |
| **Methods** | API Keys, OAuth 2.0, JWT, SSO | RBAC, ABAC, Scopes, Multi-tenant checks |

### Multi-Tenant Architecture Diagram
```text
[User A] --(Token A)--> [API Gateway] 
                           |--> AuthN: Token Valid? (Yes)
                           |--> AuthZ: Can access Thread 123? (Yes)
                           |--> [LangGraph Deployment] --> (Thread 123)

[User B] --(Token B)--> [API Gateway]
                           |--> AuthN: Token Valid? (Yes)
                           |--> AuthZ: Can access Thread 123? (No, B owns Thread 456)
                           |--> [403 Forbidden]
```

### RBAC Role Examples
- **Admin:** Can create assistants, view all threads, delete threads.
- **User:** Can create threads, view only their own threads, run on their own threads.

## 8. Deep Dive: Storage Durability

**Durability** means data is safely written to disk and won't be lost if the server crashes.

### Disaster Recovery and Backups
- **Checkpoints:** Stored in Postgres under the hood. LangGraph Cloud handles backups automatically.
- **TTL Policies:** Implement Time-To-Live logic if you must delete user data after 30 days for compliance.
- **Cross-Region Replication:** In Self-Hosted setups, ensure your Postgres database replicates across availability zones.

### Checkpoints and Memory Stores
- **Thread Checkpoints:** Store the conversation graph state (up to 25MB).
- **Memory Store:** A persistent key-value store for cross-thread data (e.g., user preferences).
  
```python
# Accessing memory store within a node
def my_node(state, config, store):
    # Retrieve user preference
    pref = store.get("user_123", "preferences")
    return {"messages": [f"I remember you like {pref.value['color']}!"]}
```

## 9. Deep Dive: langgraph.json and Deployment Config

The `langgraph.json` file is the entry point for deployment.

**Schema Example:**
```json
{
  "dependencies": ["."],
  "python_version": "3.11",
  "graphs": {
    "my_agent": "./src/agent.py:graph_app"
  },
  "env": ".env"
}
```

## 10. Hands-On Projects

### Project D.1: Write a complete `langgraph.json`
**Objective:** Create a valid config for a multi-agent system.
**Step-by-step:**
1. Create a `langgraph.json` file.
2. Define two graphs in the `"graphs"` block: `"researcher": "agents/research.py:app"` and `"writer": "agents/write.py:app"`.
3. Set `python_version` to "3.12".

### Project D.2: Deploy an agent & create revisions
**Objective:** Understand the deployment lifecycle.
**Step-by-step:**
1. Run `langgraph deploy` in your CLI.
2. Note the initial revision.
3. Change a print statement in your code and run `langgraph deploy` again. Observe the new revision created.

### Project D.3: Create 2 assistants
**Objective:** Use configurations to create product variants.
**Step-by-step:**
1. In the LangSmith UI, go to your deployed graph.
2. Create Assistant 1 with `config={"configurable": {"model": "gpt-3.5-turbo"}}`.
3. Create Assistant 2 with `config={"configurable": {"model": "gpt-4o"}}`.

### Project D.4: Configure API Key Authentication
**Objective:** Secure your deployment behind API keys.
**Step-by-step:**
1. In LangGraph Cloud, go to Settings > Auth.
2. Generate a new API Key for your deployment.
3. In your client script, instantiate `get_client(api_key="YOUR_KEY")`.
**Expected Output:** Unauthenticated requests fail with 401; authenticated requests succeed.

### Project D.5: Manage Cross-Thread Memory
**Objective:** Use the Memory Store to persist user preferences.
**Step-by-step:**
1. Add a `store` parameter to your node.
2. Write a function that saves a user's name to the store based on a fixed `user_id`.
3. Create two different threads for the same user.
4. Verify that the agent can recall the user's name in the second thread.
**Learning Goal:** Understand the difference between thread-local state and global cross-thread memory.

## 11. Exam Prep: Key Takeaways

### Terminology to Memorize
- **Graph:** Code blueprint.
- **Deployment:** Live instance.
- **Revision:** Code version.
- **Assistant:** Configured variant.
- **Thread:** User session.
- **Run:** Single execution.

### Common Exam Patterns & Edge Cases
| Question Pattern | Answer / Rule | Edge Case / Gotcha |
| :--- | :--- | :--- |
| **"Which hosting model for HIPAA?"** | Hybrid or Self-Hosted | Cloud is not strictly isolated. |
| **"How to update model without code change?"** | Create a new Assistant | Re-deploying is unnecessary. |
| **"Agent lost context after a week. Why?"** | Thread was deleted or TTL expired | Threads aren't infinite. |
| **"User A sees User B's data. Why?"** | AuthZ issue or shared thread_id | Always bind 1 thread to 1 user. |
| **"State size exceeds 25MB limit."** | Graph crashes on save | Implement a summarization node. |

## 12. Resources
- [LangSmith Deployment Foundations](https://academy.langchain.com/courses/langsmith-deployment)
- LangGraph Cloud Official Documentation
- LangSmith Platform Docs
