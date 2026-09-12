# Section 2: Test - Building Reliable Agents

## 1. Chapter Overview

Welcome to Section 2 of the LCAE (LangChain Certified Agent Engineer) study guide. This chapter covers the **Test** section of the exam, which accounts for **25% (10 questions)** of your score. 

This section focuses entirely on how to evaluate, test, and monitor agents using LangSmith and LangChain evaluation frameworks. The official recommended prerequisite is the [Building Reliable Agents](https://academy.langchain.com/courses/building-reliable-agents) course.

### What this section covers and why it matters
Testing generative AI applications is fundamentally different from traditional software testing. In traditional software, `2 + 2` always equals `4`. You write an `assert(result == 4)` and move on. 

LLMs, however, are non-deterministic. The same prompt can yield different phrasing, different reasoning paths, or entirely different answers across runs. Furthermore, an agent may correctly execute a multi-step plan but still provide a suboptimal final answer. Testing these systems requires statistical approaches, domain-specific evaluation metrics, and a mindset shift from binary correctness to a spectrum of quality.

This section covers the tools and methodologies required to test AI systems robustly, systematically, and continuously.

### PM Perspective: Quality Assurance is NOT Optional for AI Products
For a Technical Product Manager (PM), testing AI is your primary defense against product risk. 

If a traditional feature breaks, the button doesn't work. If an AI feature breaks, the AI might insult a customer, leak proprietary information, hallucinate a fake company policy, or cost thousands of dollars in a runaway loop. 

QA is no longer just a phase at the end of the sprint; it is the core bottleneck of shipping AI products. If you cannot measure agent quality, you cannot improve it. If you cannot evaluate an agent offline, you cannot safely deploy it online. Your role is to define what "good" looks like, set the acceptable thresholds for launch, and ensure the engineering team is measuring against the right criteria.

### The Fundamental Challenge: Agents are Non-Deterministic
Because agents operate using large language models, their outputs are stochastic (random). 
* An agent might take 3 tool calls today and 4 tomorrow for the same user request.
* The latency might vary based on the model provider's load.
* The output format might shift if the model disobeys instructions.
* Upgrading a model (e.g., from GPT-4-turbo to GPT-4o) changes the behavior of your entire system.

You cannot test this with just 5 manual prompts. You need datasets, continuous evaluation, programmatic checks, and LLMs grading other LLMs.

---

## 2. Mental Models for PMs

To effectively lead AI product development, you need mental models that map these new testing paradigms to familiar concepts.

| AI Testing Concept | Traditional Counterpart / Analogy | Explanation |
| :--- | :--- | :--- |
| **Testing Agents** | QA for a Call Center | You cannot script every interaction a call center agent will have. Instead, you give them guidelines, listen to call recordings, grade them on a rubric (politeness, accuracy, resolution), and measure aggregate quality. |
| **Code-Based Evaluators** | Hard Requirements (Must have degree) | Binary, deterministic checks. Does the output contain a URL? Is it less than 500 characters? Is the JSON valid? It’s a fast, cheap resume filter. |
| **LLM-as-Judge Evaluators** | Soft Skills Assessment (Interview) | Nuanced, qualitative checks. Is the tone professional? Did it answer the user's underlying intent? This requires a "smart" judge but takes more time and money. |
| **Datasets** | Test Cases / Regression Suite | A static list of inputs and expected outputs (or expected behaviors). You run your agent against these before any release to ensure you haven't broken existing functionality. |
| **Experiments** | A/B Tests | Running a new prompt, model, or architecture against your dataset and comparing the evaluation scores to the previous baseline to prove it's actually better. |
| **Offline Evals** | Pre-Release Testing (Beta / Staging) | Running tests in the lab on static data before users ever see the new version. |
| **Online Evals** | Production Monitoring (Datadog/NewRelic) | Grading real user interactions in the wild. Did the user actually get help? Was the live interaction safe? |

---

## 3. First Principles: Requirements & Solution Design

### What does 'reliable' mean for an agent?
Before writing a single test, PMs must define what "reliable" means for their specific use case. Reliability is a composite metric made up of several dimensions:

1. **Correctness (Accuracy):** Does the agent provide factually correct information based on its context?
2. **Consistency:** Does the agent provide the same quality of answer across multiple identical or similar queries?
3. **Safety (Guardrails):** Does the agent refuse to answer malicious queries? Does it avoid toxic output?
4. **Adherence:** Does it follow the formatting rules (e.g., JSON only) and business rules (e.g., never offer discounts > 10%)?
5. **Cost:** Does the agent solve the problem within an acceptable token budget?
6. **Latency:** Does the agent respond within the user's patience threshold?

### The Evaluation Pyramid
Similar to the traditional testing pyramid (Unit > Integration > E2E), agent testing has a hierarchy:

1. **Production Monitoring (Top):** Online evaluators running on live traffic. High volume, real data, helps detect drift.
2. **End-to-End Evals (Middle):** Offline evaluators running large datasets. Tests the entire system (retrieval + agent routing + generation).
3. **Component Tests (Bottom):** Unit testing individual pieces. (e.g., Evaluating just the retrieval step, or just the tool-calling format).

### Requirements Template for Agent Quality
When writing product requirements for an AI agent, include a section like this:

> **Quality Acceptance Criteria:**
> - **Dataset:** Agent must be evaluated against the `v2_customer_support` dataset (500 examples).
> - **Code-based Metric:** Must achieve 100% pass rate on JSON schema validation.
> - **LLM-as-Judge Metric:** Must achieve > 4.2/5 average score on Helpfulness (graded by GPT-4o).
> - **Safety:** Must trigger refusal on 100% of the `adversarial_injections_v1` dataset.
> - **Performance:** P90 latency must be under 4 seconds. Average cost per run under $0.05.

### Cost-Quality Tradeoff Framework
Better evaluation costs more money. If you evaluate a 100,000-user app with an LLM-as-judge on every single query, your evaluation cost might exceed your feature cost.
* **Low Risk / High Volume:** Use Code-based evaluators or sample 1% of traffic for LLM-as-judge.
* **High Risk / Low Volume:** Use LLM-as-judge on 100% of traffic, maybe even ensemble (multiple LLM judges).

---

## 4. Deep Dive: Code-Based Evaluators vs LLM-as-Judge

This is a massive topic on the exam. You must know exactly when to apply each type of evaluator.

### 4.1 Code-Based Evaluators

**What they are:** 
Deterministic Python functions that take the agent's output and return a score (usually 0 or 1, or a continuous math metric) based on programmable logic. 

**Types of Code-Based Evals:**
* **Exact Match:** `output == expected_output` (Rarely useful for raw text, useful for IDs or categories).
* **Regex Match:** Does the output contain a valid SSN format?
* **JSON Schema Validation:** Can the output be parsed by `json.loads` and match Pydantic?
* **String Containment:** Does the output contain the required legal disclaimer?
* **Length/Latency:** Is the output under 500 words? Did it take less than 3 seconds?

**Pros:** 
Extremely fast, cheap (free compute), 100% deterministic, highly reproducible.
**Cons:** 
Brittle. If the agent says "Here is the URL: http..." instead of just the URL, a regex might fail. Cannot assess tone, helpfulness, or semantic truth.

**Code Example (LangSmith SDK):**
```python
from langsmith.schemas import Run, Example

def exact_match_evaluator(run: Run, example: Example) -> dict:
    # Extract the predicted string from the run
    prediction = run.outputs.get("output", "")
    # Extract expected string from the dataset example
    expected = example.outputs.get("expected", "")
    
    score = 1 if prediction.strip() == expected.strip() else 0
    return {"key": "exact_match", "score": score}

def length_evaluator(run: Run, example: Example) -> dict:
    prediction = run.outputs.get("output", "")
    # Penalize if it's too long
    score = 1 if len(prediction) < 1000 else 0
    return {"key": "length_under_1000", "score": score}
```

### 4.2 LLM-as-Judge Evaluators

**What they are:**
Using an LLM (often a more capable, more expensive one, like GPT-4o or Claude 3.5 Sonnet) to grade the output of your agent by providing it with a grading rubric in the prompt.

**Types of LLM-as-Judge Evals:**
* **Correctness:** Given the expected answer and the actual answer, are they factually equivalent?
* **Helpfulness:** Did this actually solve the user's problem?
* **Faithfulness (Hallucination check):** Is the answer fully supported by the retrieved context?
* **Relevance:** Does the answer directly address the prompt?
* **Safety/Toxicity:** Is the language appropriate?

**Pros:**
Understands semantic meaning, handles varying phrasing, can make qualitative judgments, robust to minor formatting issues.
**Cons:**
Expensive (costs tokens to grade tokens), slower, non-deterministic (judge might give 4/5 today and 5/5 tomorrow), susceptible to LLM biases (like preferring longer answers).

**Code Example (LangChain/LangSmith):**
```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

# 1. Define the desired output structure of the judge
class Grade(BaseModel):
    score: int = Field(description="Score from 1 to 5 based on helpfulness")
    reasoning: str = Field(description="Why this score was given")

# 2. Create the evaluator function
def helpfulness_evaluator(run: Run, example: Example) -> dict:
    user_input = example.inputs.get("input")
    prediction = run.outputs.get("output")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(Grade)
    
    prompt = PromptTemplate.from_template(
        "Evaluate the assistant's response for helpfulness.\n"
        "User query: {query}\n"
        "Assistant response: {response}\n"
        "Give a score from 1 (unhelpful) to 5 (extremely helpful)."
    )
    
    # Run the judge LLM
    chain = prompt | structured_llm
    result = chain.invoke({"query": user_input, "response": prediction})
    
    # Normalize score to 0.0 - 1.0 for LangSmith
    normalized_score = result.score / 5.0
    
    return {
        "key": "helpfulness",
        "score": normalized_score,
        "comment": result.reasoning
    }
```

### Comparison Table for PMs

| Dimension | Code-Based Evaluators | LLM-as-Judge Evaluators |
| :--- | :--- | :--- |
| **Speed** | Milliseconds | Seconds |
| **Cost** | Essentially Free | High (Token costs) |
| **Determinism** | 100% | ~90-95% (depends on temp/model) |
| **Semantic Understanding** | None | High |
| **Best Used For** | Format, Length, JSON, PII masking | Tone, Accuracy, Context, Helpfulness |
| **PM Analogy** | KPI / Metric (Bounce rate, latency) | NPS / CSAT / Qualitative Feedback |

**Decision Framework:**
1. Start with code-based evaluators wherever possible.
2. For subjective quality, use LLM-as-judge.
3. Combine them! E.g., The output MUST pass the JSON code-evaluator, AND it MUST score > 4 on the LLM-as-judge Helpfulness scale.

---

## 5. Deep Dive: Online vs Offline Evaluators

You must know the difference between running evaluations in the lab vs running them in production.

### 5.1 Offline Evaluators (Pre-release)
Offline evaluators are run during development and CI/CD pipelines. 
You run your agent over a static **Dataset** of known inputs, wait for all outputs to generate, and then run evaluators.

**Key Characteristics:**
* **Environment:** Lab environment, controlled.
* **Inputs:** Fixed, pre-selected datasets.
* **Expected Outputs:** Usually known (you have a ground-truth "reference answer" in the dataset).
* **Use Cases:** Regression testing, comparing Model A vs Model B, prompt engineering iteration.
* **Workflow:** Modify Code -> Run Experiment on Dataset -> Compare to previous runs.

### 5.2 Online Evaluators (Post-release)
Online evaluators are run in production on live user traffic.
Since you don't know what users will type, you *cannot* have a "ground truth" reference answer. The evaluator must judge the interaction on its own merits (e.g., "Did the agent sound polite?", "Did the agent crash?").

**Key Characteristics:**
* **Environment:** Production.
* **Inputs:** Unpredictable, real user data.
* **Expected Outputs:** Unknown.
* **Use Cases:** Detecting model drift, monitoring toxicity, capturing bad UX live.
* **Workflow:** User chats -> Trace logged to LangSmith -> Async background job runs evaluator -> Updates LangSmith dashboard.

**Managing Cost for Online Evals:**
Running GPT-4o as a judge on 100% of your production traffic is prohibitively expensive. You use **Sampling**.
* Sample 5% of traces for Helpfulness.
* Sample 100% of traces using a cheap code-based evaluator (e.g., regex for PII leaks).
* Trigger LLM-as-judge only when specific conditions are met (e.g., user hits a 'thumbs down' button).

### Comparison Table for PMs

| Feature | Offline Evaluators | Online Evaluators |
| :--- | :--- | :--- |
| **Timing** | Before deployment | After deployment |
| **Ground Truth** | Usually available | Never available |
| **Focus** | Regression, Benchmarking, A/B Testing | Drift, Quality Monitoring, Alerting |
| **Cost Control** | Run once per experiment | Use sampling to control costs |
| **PM Analogy** | Pre-launch QA and Beta Testing | Datadog/Amplitude Post-launch Analytics |

---

## 6. Deep Dive: Running and Interpreting Experiments

In LangSmith, an "Experiment" is a specific run of a specific agent version against a specific dataset, evaluated by specific evaluators.

### The Experiment Workflow
1. **Dataset:** You create a collection of `inputs` and optionally `expected_outputs`.
2. **Target (Agent):** You define a function that takes the `input` and returns an `output`.
3. **Evaluators:** You define functions that take `(run, example)` and return scores.
4. **Execution:** You run the target over all dataset items, then run evaluators on the results.

### Code Example: Full Experiment Workflow
```python
from langsmith import Client, evaluate

client = Client()

# 1. Ensure dataset exists
dataset_name = "Customer_Support_QAs"
# (Assume dataset is already populated in LangSmith UI or via code)

# 2. Define target agent function
def my_agent_target(inputs: dict) -> dict:
    user_query = inputs["question"]
    # ... call your LangChain/LangGraph agent ...
    result = agent.invoke({"messages": [("user", user_query)]})
    return {"output": result["messages"][-1].content}

# 3. Define evaluators
evaluators = [exact_match_evaluator, helpfulness_evaluator] # from earlier

# 4. Run Experiment
experiment_results = evaluate(
    target=my_agent_target,
    data=dataset_name,
    evaluators=evaluators,
    experiment_prefix="GPT-4o_v2_prompt",
    metadata={"version": "2.0", "model": "gpt-4o"}
)
```

### Interpreting Results (PM Lens)
When the experiment finishes, you look at the LangSmith UI. You will see:
* **Aggregate Scores:** "Helpfulness: 85% | Exact Match: 42%". These are your top-line KPIs.
* **Per-Example Breakdowns:** You can click into the 15% of examples that failed helpfulness.
* **Comparison View:** You can compare "GPT-4o_v2_prompt" against "Claude3_v1_prompt" side-by-side. 
* **Statistical Significance:** PMs must ensure the dataset is large enough. If you test on 5 examples, a change in 1 score swings the average by 20%. Do not make architectural decisions on tiny datasets.

---

## 7. Deep Dive: Evaluator Alignment

**The Meta-Problem:** If an LLM is judging your agent, who judges the judge? 
How do you know that when your GPT-4o judge says "5/5 - Helpful", a real human user would agree?

This is called **Evaluator Alignment**. You are aligning your automated metric with human preference.

### Techniques for Evaluator Alignment
1. **Human Annotation:** A human (PM, domain expert) goes through 100 outputs and manually scores them 1-5. You then run your LLM evaluator on those same 100 outputs. You calculate the correlation between the human scores and LLM scores. If they match 90% of the time, your evaluator is aligned.
2. **Evaluator Calibration:** If the LLM is consistently grading too harshly, you modify the prompt of the judge (e.g., adding explicit examples of what constitutes a 3 vs a 4) to calibrate it.
3. **Inter-rater Reliability:** Using multiple judges (e.g., GPT-4o and Claude) and checking if they agree with each other.

### Common Pitfalls
* **Evaluator Gaming:** An LLM judge might just reward longer answers ("verbosity bias"). The agent learns to just output massive blocks of text to get high scores, even if it's annoying to the user.
* **Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure." 

**PM Lens:** Evaluator alignment is validating your success metrics. If your dashboard says "100% Customer Satisfaction" but your users are churning, your metric is wrong. Calibrate the evaluator.

---

## 8. Deep Dive: Adding Examples to a Dataset

A dataset is only as good as the examples within it. Over time, your datasets must grow.

### Sources of Examples
1. **Manual Creation:** PMs and domain experts sit down and write the top 20 questions they expect users to ask, along with the perfect reference answers.
2. **From Production Traces:** You look at LangSmith logs. You find a real user who asked a great question. You click "Add to Dataset" to promote that trace into a test case.
3. **Failure Analysis (Regression):** A user reports a bug where the agent hallucinated. You fix the bug. You immediately add that user's interaction to the dataset so you can mathematically prove the bug never comes back.
4. **Synthetic Generation:** You use an LLM to generate 1,000 variations of user queries to stress-test your system rapidly.
5. **Edge Cases:** Adversarial prompt injections, queries in foreign languages, completely empty inputs.

### Best Practices for Dataset Design
* **Diversity:** Don't just test 50 variations of "reset my password". Test the long tail of edge cases.
* **Versioning:** Keep datasets tagged (e.g., `prod-dataset-v1`).
* **Categorization:** Use Key-Value tags in LangSmith to categorize data (e.g., `intent: refund`, `difficulty: hard`). This allows you to see experiment results broken down by category.

**Code Example: Adding Examples Programmatically**
```python
from langsmith import Client

client = Client()
dataset_name = "Refund_Edge_Cases"

# Create dataset if it doesn't exist
if not client.has_dataset(dataset_name=dataset_name):
    client.create_dataset(dataset_name=dataset_name, description="Tricky refund cases")

# Add a specific edge case
client.create_example(
    inputs={"question": "I bought this 3 years ago but it broke today, refund pls?"},
    outputs={"expected_behavior": "Politely refuse, outside of 30 day window."},
    dataset_name=dataset_name
)
```

---

## 9. Hands-On Projects

These projects are designed to cement the technical implementation and the PM theory.

### Project T.1: Create a baseline dataset
**Objective:** Create a representative dataset for a support agent.
**Steps:**
1. Define 5 clear, happy-path queries (e.g., "What are your hours?").
2. Define 5 edge-case queries (e.g., "WHAT ARE YOUR HOURSSSS?!").
3. Define 5 out-of-scope queries (e.g., "How do I fix my car?").
4. Upload all 15 to a LangSmith dataset using the Python SDK.
**Learning Goal:** Understand the structure of `inputs` and `outputs` in LangSmith datasets.

### Project T.2: Write Code-Based Evaluators
**Objective:** Create fast, deterministic checks.
**Steps:**
1. Write a Python function that asserts the output length is < 200 characters.
2. Write a Python function that uses regex to ensure NO email addresses are in the output (PII check).
3. Test these functions against dummy strings.
**Learning Goal:** Experience how fast and rigid code evals are.

### Project T.3: Write LLM-as-Judge Evaluators
**Objective:** Create semantic checks.
**Steps:**
1. Write a LangChain PromptTemplate that asks an LLM to grade output safety (1-5).
2. Wire it up using `.with_structured_output()` to force JSON extraction.
3. Test it by feeding it a toxic string and ensuring it scores a 1.
**Learning Goal:** Learn prompt engineering for evaluator LLMs.

### Project T.4: Run a Full Experiment
**Objective:** Compare two configurations.
**Steps:**
1. Run an agent with `temperature=0` against your Project T.1 dataset using `evaluate()`.
2. Run the same agent with `temperature=1` against the dataset.
3. Open LangSmith UI and compare the scores.
**Learning Goal:** Understand the iterative A/B testing workflow.

### Project T.5: Online Evaluation
**Objective:** Grade live traffic async.
**Steps:**
1. Write a script that pulls a recent trace from LangSmith.
2. Run your LLM-as-judge on that trace.
3. Use the SDK to append the score as feedback to the original trace ID.
**Learning Goal:** Differentiate between `evaluate()` for offline datasets and appending feedback to live traces.

---

## 10. Exam Prep: Key Takeaways

Memorize these points for the LCAE exam:

### Summary Matrix
| Concept | Key Keyword | Main Purpose |
| :--- | :--- | :--- |
| Code Evaluator | Deterministic, Fast | Format, schemas, constraints |
| LLM-as-Judge | Semantic, Non-deterministic | Quality, tone, accuracy, relevance |
| Offline Evals | Datasets, Experiments | Pre-deployment regression/benchmarking |
| Online Evals | Traces, Live Traffic | Post-deployment monitoring/drift |
| Evaluator Alignment | Human vs LLM | Ensuring the judge is grading correctly |

### Common Exam Question Patterns
* **"You need to check if the agent outputs valid JSON. Cost is a concern. What do you use?"** -> *Code-based evaluator (JSON parser).*
* **"You need to measure if answers are actually helpful to users. What do you use?"** -> *LLM-as-Judge.*
* **"Your team is arguing about which prompt is better. What should you do?"** -> *Create a dataset, run an experiment, and compare the empirical scores.*
* **"How do you know your LLM evaluator is accurate?"** -> *Align it against human annotations (Evaluator calibration).*

### Gotchas
* **Don't use LLM-as-judge for everything.** It's expensive and slow. If you can use a code eval, do it.
* **Datasets are not static.** A good team is constantly promoting production failures into their offline datasets.
* **`evaluate()` is for datasets.** You do not use the `evaluate()` function on live production traffic; you append feedback to traces.

---

## 11. Resources
* [Building Reliable Agents (LangChain Academy)](https://academy.langchain.com/courses/building-reliable-agents)
* [LangSmith Evaluation Documentation](https://docs.smith.langchain.com/evaluation)
* [LangChain Quickstart for Testing](https://python.langchain.com/docs/langsmith/evaluation)

*End of Chapter 2*
