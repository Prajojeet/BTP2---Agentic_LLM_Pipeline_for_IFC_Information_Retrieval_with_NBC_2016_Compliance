"""Tiny CLI: ``ifc-agent path/to/file.ifc "your question"``"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import run_query
from .compliance import FindingsStore
from .ifc_context import IFCContext


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a natural-language question about an IFC file.")
    parser.add_argument("ifc_path", help="Path to an .ifc file.")
    parser.add_argument("question", help="The natural-language question.")
    parser.add_argument("--trace", action="store_true", help="Print the full ReAct trace as JSON.")
    parser.add_argument(
        "--no-compliance",
        action="store_true",
        help="Disable NBC fire-safety compliance tools (and the findings store).",
    )
    args = parser.parse_args()

    ctx = IFCContext.from_path(args.ifc_path)
    findings_store = None if args.no_compliance else FindingsStore()
    result = run_query(ctx, args.question, findings_store=findings_store)

    print("\n=== Selected tools ===")
    print(", ".join(result["selected_tools"]) or "(none)")
    print("\n=== Answer ===")
    print(result["answer"])
    if findings_store is not None and len(findings_store):
        print("\n=== Compliance findings ===")
        for f in findings_store.all():
            print(f"  [{f.verdict.upper():<13}] {f.finding_id}  {f.check_name}")
            print(f"                 {f.clause}")
            print(f"                 {f.summary}")
    if args.trace:
        print("\n=== Trace ===")
        print(json.dumps(result["trace"], indent=2, default=str))
    if result.get("error"):
        print(f"\n[error] {result['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
