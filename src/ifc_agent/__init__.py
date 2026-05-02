"""ifc_agent — agentic NL → IFC information retrieval.

Replicates the architecture of:
    Hellin, Nousias, Borrmann (2025).
    "Natural Language Information Retrieval from BIM Models:
     An LLM-Based Agentic Workflow Approach." EC3 2025.

Two-step workflow:
    1. Chain-of-Thought tool selection
    2. ReAct execution loop on the selected subset
"""

__version__ = "0.1.0"
