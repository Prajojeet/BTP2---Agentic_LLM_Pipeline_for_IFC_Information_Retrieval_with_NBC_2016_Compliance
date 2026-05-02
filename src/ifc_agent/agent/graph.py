"""LangGraph workflow with session memory + multi-turn conversation history.

Per turn:
    [START] → select_tools (CoT) → react_execute (ReAct) → [END]

Across turns:
    Conversation history is checkpointed by ``MemorySaver`` and keyed by
    ``thread_id`` (one per Streamlit session). The selector and the executor
    both see the full prior history, so follow-up questions like
    "show me the failing rooms" carry context.

A ``FindingsStore`` (also session-scoped) is passed in by the caller and held
by the compliance tools through closures. The store is ORTHOGONAL to the
LangGraph checkpoint: checkpoint = messages, store = structured findings.
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from ..compliance import FindingsStore
from ..config import get_settings
from ..ifc_context import IFCContext
from ..tools import build_tools
from .tool_selector import select_tools

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    user_query: str
    selected_tools: list[str]
    selection_reasoning: str
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str
    error: Optional[str]


_REACT_SYSTEM_PROMPT = """You are an expert BIM analyst with knowledge of the National \
Building Code of India 2016, Part 4 (Fire and Life Safety). The user has uploaded an IFC \
building model and is asking questions about it across a multi-turn conversation.

You have two kinds of tools:

A. RETRIEVAL & GEOMETRY tools (the 29 IFC tools) — read the model: classes, properties, \
quantities, geometry, containment.

B. FIRE-SAFETY COMPLIANCE tools (NBC 2016 Part 4) — perform structured checks (means of \
egress, travel distance, exit widths, dead-end corridors, compartmentation, refuge area). \
Each compliance tool returns a Finding {finding_id, verdict, clause, summary, failures} \
which is automatically saved to session memory. You can recall findings later with \
`list_compliance_findings` and drill into one with `get_finding_details`.

Workflow per turn:
1. THINK about whether the question is asking for raw model data OR a code-compliance verdict.
2. CALL the appropriate tool(s). Compliance checks should usually be called once and then \
referenced by `finding_id`.
3. OBSERVE the JSON each tool returns.
4. REPEAT only if a follow-up call is genuinely needed.
5. Give a concise, accurate final answer. For compliance results, ALWAYS quote the \
finding_id and the NBC clause so the user can audit.

Hard rules:
- Work from data the tools return; never invent IDs, names, numbers, or NBC clauses.
- All numeric values returned by tools are in SI units (m, m², m³) unless noted.
- If a tool returns verdict='indeterminate', say so plainly — do not pretend to a verdict.
- For follow-up questions (e.g. "which rooms failed?"), prefer `get_finding_details` over \
re-running the original check.
"""


# ---------------------------------------------------------------------------
# Node 1 — CoT tool selection
# ---------------------------------------------------------------------------
def _make_selector_node(settings, has_compliance: bool):
    def selector_node(state: AgentState) -> dict:
        # Use the latest human turn as the query (history may already exist)
        msgs = state.get("messages", [])
        query = state.get("user_query") or _last_human(msgs) or ""
        result = select_tools(
            query,
            model=settings.model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
            include_compliance=has_compliance,
        )
        logger.info("Selected %d tools: %s", len(result.selected_tools), result.selected_tools)
        return {
            "selected_tools": result.selected_tools,
            "selection_reasoning": result.reasoning,
        }

    return selector_node


def _last_human(msgs: list[BaseMessage]) -> Optional[str]:
    for m in reversed(msgs):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            return m.content
    return None


# ---------------------------------------------------------------------------
# Node 2 — ReAct execution on the selected subset
# ---------------------------------------------------------------------------
def _make_executor_node(ctx: IFCContext, settings, findings_store: Optional[FindingsStore]):
    all_tools = build_tools(ctx, findings_store=findings_store)
    tools_by_name = {t.name: t for t in all_tools}

    def executor_node(state: AgentState) -> dict:
        selected_names = state.get("selected_tools") or []
        chosen = [tools_by_name[n] for n in selected_names if n in tools_by_name]
        if not chosen:
            chosen = all_tools

        llm = ChatOpenAI(
            model=settings.model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key,
        )
        agent = create_react_agent(llm, chosen, prompt=_REACT_SYSTEM_PROMPT)

        # Pass the full message history so the ReAct agent has multi-turn context.
        history = state.get("messages", [])
        if not history and state.get("user_query"):
            history = [HumanMessage(content=state["user_query"])]

        try:
            result = agent.invoke(
                {"messages": history},
                config={"recursion_limit": settings.max_iterations * 2},
            )
        except Exception as exc:
            logger.exception("ReAct agent failed: %s", exc)
            return {
                "final_answer": f"The agent failed during execution: {exc}",
                "error": str(exc),
                "messages": [],
            }

        msgs = result.get("messages", [])
        # `add_messages` dedupes by message id, so returning the full list is safe and
        # robust across LangGraph minor versions that may or may not echo the input
        # history back. Existing messages keep their ids; new ones get appended.
        final = _extract_final_answer(msgs)
        return {"messages": msgs, "final_answer": final}

    return executor_node


def _extract_final_answer(msgs: list[BaseMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            return m.content.strip()
    return "(no answer produced)"


# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------
def build_workflow(
    ctx: IFCContext,
    findings_store: Optional[FindingsStore] = None,
    checkpointer: Optional[Any] = None,
):
    """Compile a LangGraph workflow bound to a single IFC file.

    ``findings_store`` enables compliance tools when given.
    ``checkpointer`` enables multi-turn memory; pass a ``MemorySaver`` from the
    UI layer so it lives for the full Streamlit session.
    """
    settings = get_settings()
    has_compliance = findings_store is not None

    g = StateGraph(AgentState)
    g.add_node("select_tools", _make_selector_node(settings, has_compliance))
    g.add_node("react_execute", _make_executor_node(ctx, settings, findings_store))
    g.add_edge(START, "select_tools")
    g.add_edge("select_tools", "react_execute")
    g.add_edge("react_execute", END)

    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def run_query(
    ctx: IFCContext,
    user_query: str,
    *,
    findings_store: Optional[FindingsStore] = None,
    checkpointer: Optional[Any] = None,
    thread_id: str = "default",
) -> dict[str, Any]:
    """Convenience wrapper — fresh checkpointer per call (no persistence).

    For multi-turn use, build the workflow once with a long-lived checkpointer
    and call ``graph.invoke`` directly with the same ``thread_id``. The Streamlit
    app does this; ``run_query`` is a single-shot convenience for the CLI.
    """
    graph = build_workflow(ctx, findings_store=findings_store, checkpointer=checkpointer)
    state: AgentState = {
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
    }
    out = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    return {
        "answer": out.get("final_answer", "(no answer)"),
        "selected_tools": out.get("selected_tools", []),
        "selection_reasoning": out.get("selection_reasoning", ""),
        "trace": _summarise_trace(out.get("messages", [])),
        "error": out.get("error"),
    }


def _summarise_trace(msgs: list[BaseMessage]) -> list[dict]:
    """Compact, JSON-friendly trace of the ReAct loop for the UI."""
    trace: list[dict] = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            trace.append({"role": "user", "content": m.content})
        elif isinstance(m, SystemMessage):
            continue
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if isinstance(m.content, str) and m.content:
                entry["content"] = m.content
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
                ]
            trace.append(entry)
        elif isinstance(m, ToolMessage):
            trace.append(
                {
                    "role": "tool",
                    "name": m.name,
                    "content": _truncate_content(m.content),
                }
            )
    return trace


def _truncate_content(content: Any, max_chars: int = 4000) -> Any:
    if isinstance(content, str) and len(content) > max_chars:
        return content[:max_chars] + f"... [truncated, original {len(content)} chars]"
    return content
