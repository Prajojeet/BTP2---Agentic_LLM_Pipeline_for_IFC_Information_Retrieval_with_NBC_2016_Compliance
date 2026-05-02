"""Chain-of-Thought tool selection step.

Step 1 of Hellin et al. (2025): a single LLM call sees the user query + a
short catalogue of all available tools (name + one-sentence description) and
returns the subset relevant to the query. The ReAct executor then sees only
the picked subset, keeping its context window short.

The selector now also picks compliance tools when the question is about
fire safety / egress / NBC code conformance.
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..tools import tool_catalog

logger = logging.getLogger(__name__)


class ToolSelection(BaseModel):
    """Structured-output schema for the selector."""

    reasoning: str = Field(
        description="Brief, step-by-step reasoning that explains *why* these tools were chosen."
    )
    selected_tools: list[str] = Field(
        description="The names of the tools (subset of the catalogue) that the ReAct agent should be given."
    )


_SELECTOR_SYSTEM_PROMPT = """You are an expert BIM analyst. Your job is to pick the smallest sufficient \
subset of tools that a ReAct agent will need to answer a natural-language question \
about an IFC building model.

Hard rules:
1. ONLY pick tool names that appear in the catalogue below. Do not invent names.
2. Pick the SMALLEST set that is still sufficient. Fewer tools = better accuracy.
   Typical good answers contain 2–6 tools. Up to 8 is fine for genuinely complex queries.
3. Element-by-GlobalId questions usually need `get_element_properties` and/or geometry tools.
4. Counts, areas, volumes, lengths → matching Quantity Computation tools.
5. Rooms / storeys / "which X is in Y" → at least one Geometric Processing tool.
6. Fire safety, egress, exit width, travel distance, fire rating, NBC compliance → \
pick from the Fire Safety Compliance tools. Add `list_compliance_findings` or \
`get_finding_details` if the user is asking about something already checked.
7. If the question asks "what did we already check" or "show me the failing rooms / details" — \
prefer the Compliance Memory tools over re-running expensive checks.
8. Always populate `reasoning` step-by-step before listing the picks.

Catalogue (one tool per line, format: NAME [CATEGORY] — description):
{catalogue}
"""


def _format_catalog(include_compliance: bool) -> str:
    lines = []
    for entry in tool_catalog(include_compliance=include_compliance):
        lines.append(f"- {entry['name']} [{entry['category']}] — {entry['description']}")
    return "\n".join(lines)


def select_tools(
    user_query: str,
    *,
    model: str,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    include_compliance: bool = True,
) -> ToolSelection:
    """Run the CoT selector. Always returns a valid ToolSelection.

    Falls back to a safe default if the LLM call or structured-parse fails.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    structured = llm.with_structured_output(ToolSelection)

    system = _SELECTOR_SYSTEM_PROMPT.format(catalogue=_format_catalog(include_compliance))

    try:
        result = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=user_query)]
        )
        valid_names = {e["name"] for e in tool_catalog(include_compliance=include_compliance)}
        cleaned = [n for n in result.selected_tools if n in valid_names]
        if not cleaned:
            cleaned = _fallback_default()
            logger.warning("Selector returned no valid tools; falling back to %s", cleaned)
        return ToolSelection(reasoning=result.reasoning, selected_tools=cleaned)
    except Exception as exc:
        logger.warning("Tool selector failed (%s); using fallback default.", exc)
        return ToolSelection(
            reasoning=f"Selector raised {type(exc).__name__}; using safe default.",
            selected_tools=_fallback_default(),
        )


def _fallback_default() -> list[str]:
    """A safe, broad-but-not-everything default for when the selector errors out."""
    return [
        "get_state",
        "find_elements_by_ifc_class",
        "get_element_properties",
        "list_rooms",
        "get_storeys_names",
        "get_elements_area",
        "get_elements_volume",
        "get_containing_storey",
    ]
