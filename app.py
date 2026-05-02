"""Streamlit UI — IFC Agent (industrial-light theme, multi-turn chat).

Run with:  ``streamlit run app.py``

Architecture
------------
- One Streamlit session  ←→  one ``thread_id`` for LangGraph's MemorySaver.
- One uploaded IFC       ←→  one ``IFCContext`` cached for the session.
- One ``FindingsStore``  per session, shared with the compliance tools.
- Conversation is rendered as chat bubbles; per-turn we expose the picked
  tools and the ReAct trace inside collapsible details.
"""
from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

import streamlit as st
from langchain_core.messages import HumanMessage

from ifc_agent.agent.graph import _summarise_trace, build_workflow
from ifc_agent.compliance.findings_store import Finding, FindingsStore
from ifc_agent.config import get_settings
from ifc_agent.ifc_context import IFCContext

# LangGraph checkpointer — kept per-session
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ifc-agent.app")


# =============================================================================
# Page & styling
# =============================================================================
st.set_page_config(
    page_title="IFC Agent — BIM Query & NBC Compliance",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_css() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


_inject_css()


# =============================================================================
# Session-state init
# =============================================================================
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("checkpointer", MemorySaver())
    ss.setdefault("findings_store", FindingsStore())
    ss.setdefault("ifc_ctx", None)
    ss.setdefault("ifc_filename", None)
    ss.setdefault("ifc_temp_path", None)
    ss.setdefault("workflow", None)
    ss.setdefault("turns", [])  # list of dicts: {role, content, meta}


_init_state()


# =============================================================================
# Hero
# =============================================================================
def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-mark">IFC</div>
          <div>
            <h1>IFC Agent</h1>
            <div class="sub">BIM Query · NBC 2016 Part 4 Compliance · Agentic LLM</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_hero()


# =============================================================================
# Sidebar
# =============================================================================
def _open_ifc(uploaded) -> None:
    """Persist the upload to a temp file and build an IFCContext + workflow."""
    suffix = Path(uploaded.name).suffix or ".ifc"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()

    ctx = IFCContext.from_path(tmp.name)
    st.session_state.ifc_ctx = ctx
    st.session_state.ifc_filename = uploaded.name
    st.session_state.ifc_temp_path = tmp.name
    # Rebuild workflow against this context
    st.session_state.workflow = build_workflow(
        ctx,
        findings_store=st.session_state.findings_store,
        checkpointer=st.session_state.checkpointer,
    )
    # Reset turns and findings for a fresh model
    st.session_state.turns = []
    st.session_state.findings_store.clear()
    # New thread = fresh checkpoint memory
    st.session_state.thread_id = uuid.uuid4().hex


def _ifc_quick_stats(ctx: IFCContext) -> dict[str, Any]:
    try:
        n_storeys = len(ctx.by_type("IfcBuildingStorey"))
        n_walls = len(ctx.by_type("IfcWall"))
        n_doors = len(ctx.by_type("IfcDoor"))
        n_windows = len(ctx.by_type("IfcWindow"))
        n_spaces = len(ctx.by_type("IfcSpace"))
        return {
            "Schema": ctx.schema,
            "Storeys": n_storeys,
            "Walls": n_walls,
            "Doors": n_doors,
            "Windows": n_windows,
            "Spaces": n_spaces,
        }
    except Exception as exc:
        logger.warning("Quick stats failed: %s", exc)
        return {"Schema": ctx.schema}


with st.sidebar:
    st.markdown('<div class="card"><div class="card-title">Model Input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload IFC file",
        type=["ifc"],
        help="Drop any .ifc model. The agent loads it once per session.",
        label_visibility="collapsed",
    )
    if uploaded is not None and (
        st.session_state.ifc_filename != uploaded.name
    ):
        with st.spinner(f"Loading {uploaded.name}…"):
            try:
                _open_ifc(uploaded)
            except Exception as exc:
                st.error(f"Could not open IFC: {exc}")

    if st.session_state.ifc_ctx is not None:
        ctx: IFCContext = st.session_state.ifc_ctx
        st.markdown(
            f"<div style='font-family: var(--mono); font-size: 0.78rem; "
            f"margin: 6px 0 8px; color: var(--steel-700);'>📐 {st.session_state.ifc_filename}</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.ifc_ctx is not None:
        stats = _ifc_quick_stats(st.session_state.ifc_ctx)
        st.markdown('<div class="card"><div class="card-title">Model Snapshot</div>', unsafe_allow_html=True)
        cells = "".join(
            f'<div class="stat"><div class="stat-label">{k}</div>'
            f'<div class="stat-value">{v}</div></div>'
            for k, v in stats.items()
        )
        st.markdown(f'<div class="stat-grid">{cells}</div></div>', unsafe_allow_html=True)

    settings = get_settings()
    st.markdown(
        f'<div class="card"><div class="card-title">Engine</div>'
        f'<div style="font-family: var(--mono); font-size: 0.85rem;">'
        f'<div><strong>Model:</strong> {settings.model}</div>'
        f'<div><strong>Temperature:</strong> {settings.temperature}</div>'
        f'<div><strong>Max iterations:</strong> {settings.max_iterations}</div>'
        f'<div><strong>Thread:</strong> <code>{st.session_state.thread_id[:8]}…</code></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Session controls
    st.markdown('<div class="card"><div class="card-title">Session</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("New chat", use_container_width=True):
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.checkpointer = MemorySaver()
            st.session_state.turns = []
            if st.session_state.ifc_ctx is not None:
                st.session_state.workflow = build_workflow(
                    st.session_state.ifc_ctx,
                    findings_store=st.session_state.findings_store,
                    checkpointer=st.session_state.checkpointer,
                )
            st.rerun()
    with col_b:
        if st.button("Clear findings", use_container_width=True):
            n = st.session_state.findings_store.clear()
            st.toast(f"Cleared {n} findings")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# Main split — chat + compliance findings panel
# =============================================================================
left, right = st.columns([2, 1], gap="large")


# ----- Right panel: compliance findings -----
def _render_finding_card(f: Finding) -> str:
    return (
        f'<div class="finding-card {f.verdict}">'
        f'<div class="finding-id">{f.finding_id} · {f.created_at[11:19]} UTC</div>'
        f'<div class="finding-title">{f.check_name}</div>'
        f'<div class="finding-clause">{f.clause}</div>'
        f'<div class="finding-summary">{f.summary}</div>'
        f'<div style="margin-top: 8px;"><span class="verdict-pill {f.verdict}">{f.verdict}</span></div>'
        f"</div>"
    )


with right:
    st.markdown(
        '<div class="card-title" style="margin-bottom: 8px;">'
        'NBC 2016 Compliance · Session Memory</div>',
        unsafe_allow_html=True,
    )
    findings = st.session_state.findings_store.all()
    if not findings:
        st.markdown(
            '<div class="card" style="text-align: center; color: var(--concrete-700);">'
            '<em style="font-size: 0.88rem;">No compliance checks run yet.</em><br>'
            '<span style="font-size: 0.78rem;">Ask, e.g., '
            '"Check fire egress for this building" or "Verify exit widths".</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Newest first
        for f in reversed(findings):
            st.markdown(_render_finding_card(f), unsafe_allow_html=True)
            with st.expander(f"Details · {f.finding_id}", expanded=False):
                if f.failures:
                    st.caption(f"Failing elements ({len(f.failures)}):")
                    st.json(f.failures, expanded=False)
                if f.params_checked:
                    st.caption("Parameters checked:")
                    st.json(f.params_checked, expanded=False)
                if f.extras:
                    st.caption("Extras:")
                    st.json(f.extras, expanded=False)


# ----- Left panel: chat -----
def _render_chat_history() -> None:
    if not st.session_state.turns:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">▦</div>'
            "<h3>Ready when you are.</h3>"
            "<p>Upload an IFC file from the sidebar, then ask a question. "
            "The agent picks the right tools, walks the model, and — for fire-safety "
            "questions — runs NBC 2016 Part 4 compliance checks that persist in "
            "session memory.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for turn in st.session_state.turns:
        if turn["role"] == "user":
            st.markdown(
                f'<div class="chat-row user">'
                f'<div class="chat-bubble">{_escape(turn["content"])}</div>'
                f'<div class="chat-avatar">YOU</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            meta = turn.get("meta", {})
            tool_pills = ""
            if meta.get("selected_tools"):
                tool_pills = "".join(
                    f'<span class="tool-chip">{t}</span>'
                    for t in meta["selected_tools"]
                )
            st.markdown(
                f'<div class="chat-row assistant">'
                f'<div class="chat-avatar">A</div>'
                f'<div class="chat-bubble">'
                f'<div>{_escape(turn["content"])}</div>'
                f'<div class="chat-meta"><span class="pill">tools</span>{tool_pills}</div>'
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Trace expander outside the bubble for layout cleanliness
            if meta.get("trace") or meta.get("selection_reasoning"):
                with st.expander("Show reasoning & trace", expanded=False):
                    if meta.get("selection_reasoning"):
                        st.markdown("**Tool selection reasoning**")
                        st.write(meta["selection_reasoning"])
                    if meta.get("trace"):
                        st.markdown("**ReAct trace**")
                        for step in meta["trace"]:
                            role = step.get("role", "?")
                            if role == "user":
                                continue  # already shown as bubble
                            with st.container():
                                st.markdown(
                                    f'<div class="trace-step {role}">'
                                    f'<span class="role">{role}'
                                    f'{" · " + step.get("name", "") if role == "tool" else ""}'
                                    f"</span></div>",
                                    unsafe_allow_html=True,
                                )
                                if role == "assistant":
                                    if step.get("content"):
                                        st.write(step["content"])
                                    if step.get("tool_calls"):
                                        st.json(step["tool_calls"], expanded=False)
                                else:  # tool
                                    content = step.get("content")
                                    if isinstance(content, str):
                                        try:
                                            content = json.loads(content)
                                        except Exception:
                                            pass
                                    st.json(content, expanded=False)


def _escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def _run_turn(query: str) -> None:
    """Invoke the workflow for one user turn and append the assistant reply."""
    if st.session_state.workflow is None or st.session_state.ifc_ctx is None:
        st.error("Upload an IFC file first.")
        return

    st.session_state.turns.append({"role": "user", "content": query})

    with st.status("Agent at work", expanded=True) as status:
        st.write("Selecting relevant tools…")
        try:
            out = st.session_state.workflow.invoke(
                {
                    "user_query": query,
                    "messages": [HumanMessage(content=query)],
                },
                config={
                    "configurable": {"thread_id": st.session_state.thread_id},
                    "recursion_limit": get_settings().max_iterations * 2,
                },
            )
            st.write("Executing ReAct loop…")
            answer = out.get("final_answer", "(no answer produced)")
            meta = {
                "selected_tools": out.get("selected_tools", []),
                "selection_reasoning": out.get("selection_reasoning", ""),
                "trace": _summarise_trace(out.get("messages", [])),
                "error": out.get("error"),
            }
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            logger.exception("Workflow invoke failed")
            answer = f"The agent failed: {exc}"
            meta = {"selected_tools": [], "selection_reasoning": "", "trace": [], "error": str(exc)}
            status.update(label="Error", state="error", expanded=True)

    st.session_state.turns.append({"role": "assistant", "content": answer, "meta": meta})


with left:
    st.markdown('<div class="card-title">Conversation</div>', unsafe_allow_html=True)
    _render_chat_history()

    # Chat input pinned below
    if st.session_state.ifc_ctx is None:
        st.info("Upload an IFC file from the sidebar to start.")
    else:
        EXAMPLES = [
            "Give me a high-level summary of this model.",
            "How many doors are there, and what is the smallest clear width?",
            "What is the gross floor area of the building?",
            "Run a fire-egress check assuming business occupancy.",
            "Verify exit widths against NBC 2016 Part 4.",
            "Are there any rooms more than 30 m from an external exit?",
            "List every compliance finding so far.",
        ]
        with st.expander("Example questions", expanded=False):
            for ex in EXAMPLES:
                if st.button(ex, key=f"ex_{hash(ex)}", use_container_width=True):
                    _run_turn(ex)
                    st.rerun()

        prompt = st.chat_input("Ask about the model or run a compliance check…")
        if prompt:
            _run_turn(prompt)
            st.rerun()
