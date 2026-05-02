# IFC Agent — Natural-language information retrieval from IFC models

A faithful implementation of the agentic workflow.

The system lets you upload an arbitrary `.ifc` file in a Streamlit interface and ask questions about it in plain English. Behind the scenes it runs a two-step LangGraph workflow:

1. **Chain-of-Thought tool selection** — a single LLM call picks the smallest sufficient subset from a library of **29 IFCOpenShell tools** (Information Retrieval, Quantity Computation, Geometric Processing) **plus 9 fire-safety / memory tools** (NBC 2016 Part 4 compliance, beta).
2. **ReAct execution** — `langgraph.prebuilt.create_react_agent` runs the reasoning ↔ acting ↔ observing loop *with only the picked tools in its context window*.

There are 29 tools in Table 1 thay are used to derive information about the model. Here are some technical exceptional features that are State-of-the-art:

- **LangGraph.** We use LangGraph because it gives explicit, debuggable nodes that map 1-to-1 with the paper's diagram and supports MemorySaver out of the box.
- **File upload.** The app loads any IFC the user uploads through Streamlit.
- **+ NBC 2016 Part 4 fire-safety beta.** A second tool family (`check_means_of_egress`, `check_travel_distance`, `check_exit_widths`, `check_dead_end_corridors`, `check_compartmentation`, `check_refuge_area`) the selector can reach for when the question is about Indian fire-safety code conformance.
- **+ Session memory.** LangGraph's `MemorySaver` carries conversation history across turns; a separate `FindingsStore` persists structured compliance findings so follow-up questions like *"show me the rooms that failed travel distance"* don't re-run the check.

---

## 1. Quick start

```bash

# 1. Create + activate a virtualenv (Python 3.10 – 3.12)
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell

# 2. Install
pip install -e .

# 3. Set your OpenAI key
cp .env.example .env
# Open .env and paste your sk-... key
# (Optional) change IFC_AGENT_MODEL to gpt-5.4-mini, gpt-5.2, etc.

# 4. Run the UI
streamlit run app.py
```

A browser tab opens at <http://localhost:8501>. Upload an `.ifc` file in the sidebar, type a question in the main panel, hit **Run query**.

### Headless / CLI

```bash
ifc-agent path/to/model.ifc "How many doors are on the second floor?"
ifc-agent path/to/model.ifc "Total wall volume?" --trace
```

---

## 2. What you can ask

The agent works best for questions that map onto the 29 tools. Examples that should work out of the box:

| Category | Example query |
|---|---|
| Counting | *How many windows are there?* |
| Listing | *List all rooms and their long names.* |
| Quantities | *What is the total gross floor area?* |
| Quantities | *Total volume of all concrete walls?* |
| Properties | *What is the fire rating of door {GlobalId}?* |
| Spatial | *Which room is door {GlobalId} in?* |
| Spatial | *What is the floor-to-floor height between Level 1 and Level 2?* |
| Accessibility | *Are all doors at least 850 mm wide?* |
| Inspection | *Which rooms have outdoor access?* |
| Topology | *List all the elements in space {GlobalId}.* |
| **Fire safety** | *Run a fire-egress check assuming business occupancy.* |
| **Fire safety** | *Are there any rooms more than 30 m from an external exit?* |
| **Fire safety** | *Verify exit widths against NBC Part 4.* |
| **Fire safety** | *Does the building have proper fire compartmentation?* |
| **Memory** | *List every compliance finding so far.* |
| **Memory** | *Show me the failing rooms from finding F-A1B2C3D4.* |

Per the paper's results, expect ~95 % accuracy when the answer is *directly* encoded in the IFC properties, and lower (~62 %) when it requires multi-hop reasoning or geometric inference.

---

## 3. Architecture

```
                 ┌────────────────────────────┐
   user query →  │  Streamlit UI (app.py)     │
                 │  • industrial-light theme  │
                 │  • chat-style multi-turn   │
                 │  • compliance findings     │
                 │    panel                   │
                 └─────────────┬──────────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │  LangGraph workflow        │
                 │  (agent/graph.py)          │
                 │   + MemorySaver checkpoint │
                 │                            │
                 │  ┌──────────────────────┐  │
                 │  │ select_tools (CoT)   │  │   ← tool_selector.py, sees the
                 │  │  → 2-8 tool names    │  │     short catalogue of 29 + 9 tools
                 │  └──────────┬───────────┘  │
                 │             │              │
                 │             ▼              │
                 │  ┌──────────────────────┐  │
                 │  │ react_execute        │  │   ← create_react_agent with ONLY
                 │  │  (ReAct loop)        │  │     the picked tools, full prior
                 │  │                      │  │     conversation in context
                 │  └──────────┬───────────┘  │
                 │             │              │
                 └─────────────┼──────────────┘
                               │   ↘
                               │    └──→ FindingsStore (session)
                               ▼            stores Finding{verdict, clause,
                       Final NL answer       failures, params}
```

### File layout

```
ifc-agent/
├── pyproject.toml
├── requirements.txt
├── README.md
├── .env.example
├── .streamlit/config.toml              ← industrial-light theme
├── assets/styles.css                   ← drafting-grid CSS, verdict pills
├── app.py                              ← Streamlit chat UI
└── src/ifc_agent/
    ├── __init__.py
    ├── cli.py                          ← `ifc-agent` CLI
    ├── config.py                       ← env-var settings
    ├── ifc_context.py                  ← loaded-file holder
    ├── tools/
    │   ├── __init__.py                 ← build_tools(ctx, findings_store)
    │   ├── information_retrieval.py    ← 10 tools
    │   ├── quantity_computation.py     ← 8 tools
    │   └── geometric_processing.py     ← 11 tools
    ├── compliance/                     ← NEW
    │   ├── __init__.py
    │   ├── findings_store.py           ← session-scoped Finding registry
    │   └── fire_safety.py              ← 6 NBC Part 4 checks + 3 recall tools
    ├── agent/
    │   ├── __init__.py
    │   ├── tool_selector.py            ← Step 1: CoT tool selection
    │   └── graph.py                    ← Step 2: LangGraph + MemorySaver
    └── utils/
        ├── geometry.py                 ← AABB, normals, pset/qset lookups
        └── safe_serialize.py           ← JSON-safe IFC entity → dict
```

### The 29 retrieval tools

Identical names to Table 1 of the paper:

**Information Retrieval (10)** — `find_elements_by_ifc_class`, `get_element_properties`, `get_available_models`, `get_model_path`, `list_object_types_for_ifc_entity`, `list_rooms`, `get_storeys_names`, `get_type_definitions_and_instances`, `get_state`, `is_georeferenced`.

**Quantity Computation (8)** — `calculate_glazing_area`, `calculate_gross_floor_area`, `calculate_usable_floor_area`, `count_windows_on_facade`, `extract_quantity_from_property_sets`, `get_elements_area`, `get_elements_volume`, `get_pipe_length_by_type`.

**Geometric Processing (11)** — `get_element_bounding_box`, `bounding_box_intersect`, `get_containing_rooms_for_entities_type`, `get_containing_rooms_for_entity_guids`, `get_containing_storey`, `get_door_dimensions`, `get_elements_in_room`, `get_floor_to_floor_height`, `get_room_ceiling_height`, `get_rooms_with_outdoor_access`, `check_door_accessibility`.

**Geometric Processing (11)** — `get_element_bounding_box`, `bounding_box_intersect`, `get_containing_rooms_for_entities_type`, `get_containing_rooms_for_entity_guids`, `get_containing_storey`, `get_door_dimensions`, `get_elements_in_room`, `get_floor_to_floor_height`, `get_room_ceiling_height`, `get_rooms_with_outdoor_access`, `check_door_accessibility`.

### Fire-Safety Compliance (NBC 2016 Part 4, beta — 6 + 3 tools)

| Tool | NBC clause | What it checks |
|---|---|---|
| `check_means_of_egress` | 4.4.2 | Per-storey occupant load (from IfcSpace areas + occupancy class) → required vs. actual external doors |
| `check_travel_distance` | 4.5.1 / Table 9 | For every room, centroid → nearest external exit on same storey × corridor factor; flags rooms > limit |
| `check_exit_widths` | 4.5 | External-door clear widths ≥ 1.0 m, stair widths ≥ 1.5 m |
| `check_dead_end_corridors` | 4.5.4 | Heuristic: corridor-named IfcSpace > 12 m with ≤ 1 door is flagged |
| `check_compartmentation` | 3.4 | `Pset_WallCommon.FireRating` presence on walls flagged as compartmentation |
| `check_refuge_area` | 4.4.5 | Buildings > 24 m must contain at least one IfcSpace named "refuge" |
| `list_compliance_findings` | — | Recall every finding from session memory |
| `get_finding_details` | — | Drill into one finding by `F-XXXXXXXX` id |
| `clear_compliance_findings` | — | Reset session memory |

Every check returns a `Finding`:

```json
{
  "finding_id": "F-A1B2C3D4",
  "check_name": "check_travel_distance",
  "clause": "NBC 2016, Part 4, 4.5.1 / Table 9",
  "verdict": "fail",
  "summary": "23 room(s) checked against 30 m limit; 4 exceeded.",
  "params_checked": { "occupancy_class": "business", "limit_m": 30.0, "corridor_factor": 1.5 },
  "failures": [ { "element_guid": "...", "element_name": "Storage 204", "reason": "..." } ],
  "extras": { "per_room": [ ... ] },
  "created_at": "2026-05-01T..."
}
```

The finding is automatically saved to the session `FindingsStore` and rendered in the right panel of the UI as a verdict card.

> **Beta caveat.** These checks are pragmatic, not certified. Travel distance uses straight-line × corridor factor, not pathfinding. Compartmentation depends on whether the IFC carries `FireRating` data (Revit/ArchiCAD often omit it). When IFC lacks the inputs, the tool returns `verdict = "indeterminate"` and explains why — never a false pass.

### The 29 retrieval tools, common contract

Each retrieval tool:

- is a `langchain_core.tools.StructuredTool` with a Pydantic v2 args schema,
- closes over a single `IFCContext` (so the model is only opened once per file),
- returns JSON-safe dicts (no raw IFCOpenShell entities reach the LLM),
- caps output size (large lists are auto-truncated with a `truncated` flag),
- degrades gracefully when geometry / psets are missing.

---

## 4. Configuration

All behaviour lives in `.env`:

```bash
OPENAI_API_KEY=sk-...
IFC_AGENT_MODEL=gpt-5.4              # or gpt-5.4-mini, gpt-5.2, gpt-4.1
IFC_AGENT_TEMPERATURE=0
IFC_AGENT_MAX_ITERATIONS=25
```

The paper reports best results with **Claude 3.5 Sonnet**. We default to **GPT-5.4** because the user has OpenAI tokens and the architecture is model-agnostic (any chat model with tool calling will work — just change the model string).

### Optional: LangSmith tracing

If you want to inspect the full ReAct trace in LangSmith, uncomment the relevant lines in `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=ifc-agent
```

---

## 5. How the agent reasons (worked examples)

### Example A — Retrieval

**User:** *"What is the total gross floor area, and how does it split per storey?"*

**Step 1 — CoT selector picks:**
- `get_storeys_names`
- `calculate_gross_floor_area`
- `list_rooms`
- `extract_quantity_from_property_sets`

**Step 2 — ReAct loop (typical trace):**
1. Calls `calculate_gross_floor_area()` → `{"gross_floor_area_m2": 354.6, ...}`
2. Calls `get_storeys_names()` → list of storeys.
3. Calls `list_rooms()` → list of spaces, each with storey.
4. For each storey: calls `extract_quantity_from_property_sets(guid, "GrossFloorArea")` per space.
5. Sums per storey, formats final answer.

### Example B — Compliance + memory

**User turn 1:** *"Run a fire-egress check assuming business occupancy."*

The selector picks `check_means_of_egress`, `check_travel_distance`, `check_exit_widths`. Each tool returns a `Finding` (auto-saved to the store) and the agent's final answer cites the `F-XXXXXXXX` ids and clauses.

**User turn 2:** *"Show me the rooms that failed travel distance."*

The selector — *aware of the existing finding because the conversation history is in context* — picks just `get_finding_details`. The agent calls it with the right `finding_id` from turn 1 and lists the failing rooms without re-running the expensive check.

The right-hand panel of the UI renders both findings as verdict cards (green/red/amber) the whole time.

The Streamlit UI shows every reasoning step, every tool call, every JSON observation — useful both for debugging and for thesis screenshots.

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | `.env` missing | Copy `.env.example` → `.env`; paste key |
| `ifcopenshell` install error on Windows | Pip can't find a wheel | Make sure Python is **3.10 – 3.12 64-bit**; `pip install --upgrade pip setuptools` then re-run `pip install -e .` |
| Streamlit page is blank | Backend crashed | Check the terminal for tracebacks; the most common cause is an invalid `IFC_AGENT_MODEL` name |
| Tools return `null` for areas / volumes | Source IFC has no `BaseQuantities` Qto | Expected behaviour — the geometry-based fallback gives an AABB-derived estimate; the answer text usually flags this |
| `RuntimeError: not implemented` from IfcOpenShell | Schema mismatch (rare) | Open the IFC in Sortdesk / BIMcollab to confirm it is valid |
| Slow on large models | Geometry generation is expensive | The geometry settings are cached per file; the *first* geometry-using tool call warms it up |

---

## 7. Citation

If this codebase helps your thesis, please cite:

```
@inproceedings{hellin2025nl_bim,
  title     = {Natural Language Information Retrieval from BIM Models:
               An LLM-Based Agentic Workflow Approach},
  author    = {Hellin, Sylvain and Nousias, Stavros and Borrmann, Andr{\'e}},
  booktitle = {European Conference on Computing in Construction (EC3)},
  year      = {2025},
  address   = {Porto, Portugal},
}
```
