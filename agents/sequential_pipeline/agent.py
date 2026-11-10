"""
Sequential Pipeline Pattern Implementation.

Reference:
Agentic Design Patterns Handbook by Felix Williams — Chapter 5.5
"Agents are arranged in a fixed sequence, with each agent's output becoming
the next agent's input—like a manufacturing assembly line. Each agent handles
one transformation stage: Ingestion -> Extraction -> Analysis -> Formatting."

Pattern Card:
- Type: Multi-Agent, Linear Assembly Line Pipeline
- Best For: Repeatable data pipelines, document processing, compliance auditing
- Complexity: Low-Medium
- Performance Lift: High throughput, deterministic stage verification, clean checkpointing
- Key Risk: Rigidity to dynamic branch needs; slowest stage limits overall system throughput
"""

import json
from typing import Dict, Any, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from shared.utils.model_factory import get_model


class PipelineState(TypedDict):
    """Graph state tracking data transformations across assembly-line stages."""
    raw_input: str
    sanitized_text: Optional[str]
    extracted_entities: Optional[Dict[str, Any]]
    risk_evaluation: Optional[str]
    final_deliverable: Optional[str]


model = get_model()

# ==========================================
# Stage 1: Ingestion & Normalization
# ==========================================
def ingest_stage_node(state: PipelineState) -> dict:
    """Stage 1: Cleans, strips noise, and normalizes unstructured input."""
    raw = state["raw_input"]
    prompt = (
        f"You are the Data Ingestion & Sanitization Specialist.\n"
        f"Raw Unstructured Input:\n{raw}\n\n"
        "Strip irrelevant noise, normalize timestamps/formatting, and extract clean core content."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"sanitized_text": str(resp.content)}


# ==========================================
# Stage 2: Entity & Metric Extraction
# ==========================================
def extract_stage_node(state: PipelineState) -> dict:
    """Stage 2: Extracts structured entities, KPIs, and operational parameters."""
    clean_text = state.get("sanitized_text", "")
    prompt = (
        f"You are the Information Extraction Specialist.\n"
        f"Cleaned Text:\n{clean_text}\n\n"
        "Extract key entities, metrics, technical constraints, and requirements into JSON:\n"
        '{"entities": [...], "metrics": {...}, "constraints": [...]}'
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    text = str(resp.content)
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except Exception:
        data = {"summary": text[:300]}

    return {"extracted_entities": data}


# ==========================================
# Stage 3: Risk & Gap Assessment
# ==========================================
def analyze_stage_node(state: PipelineState) -> dict:
    """Stage 3: Performs strategic risk assessment and gap identification."""
    clean_text = state.get("sanitized_text", "")
    entities = state.get("extracted_entities", {})

    prompt = (
        f"You are the Strategic Risk & Quality Assessor.\n"
        f"Context:\n{clean_text}\n\n"
        f"Extracted Entities:\n{json.dumps(entities, indent=2)}\n\n"
        "Identify the top 3 architectural risks, compliance gaps, or feasibility concerns."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"risk_evaluation": str(resp.content)}


# ==========================================
# Stage 4: Executive Formatting & Packaging
# ==========================================
def format_stage_node(state: PipelineState) -> dict:
    """Stage 4: Formats findings into a production deliverable."""
    clean_text = state.get("sanitized_text", "")
    entities = state.get("extracted_entities", {})
    risks = state.get("risk_evaluation", "")

    prompt = (
        f"You are the Executive Deliverable Publisher.\n"
        f"Context:\n{clean_text}\n\n"
        f"Entities:\n{json.dumps(entities)}\n\n"
        f"Risk Audit:\n{risks}\n\n"
        "Format into an elegant, publication-ready Executive Intelligence Memorandum "
        "with key takeaways, table of metrics, and next steps."
    )
    resp = model.invoke([HumanMessage(content=prompt)])
    return {"final_deliverable": str(resp.content)}


builder = StateGraph(PipelineState)

builder.add_node("stage_1_ingest", ingest_stage_node)
builder.add_node("stage_2_extract", extract_stage_node)
builder.add_node("stage_3_analyze", analyze_stage_node)
builder.add_node("stage_4_format", format_stage_node)

builder.add_edge(START, "stage_1_ingest")
builder.add_edge("stage_1_ingest", "stage_2_extract")
builder.add_edge("stage_2_extract", "stage_3_analyze")
builder.add_edge("stage_3_analyze", "stage_4_format")
builder.add_edge("stage_4_format", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
