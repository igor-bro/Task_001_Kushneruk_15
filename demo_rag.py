"""Демо Agentic RAG (крок 8)."""

from __future__ import annotations

import sys

from knowledge import KNOWLEDGE_DOCS, get_collection, knowledge_search, search_docs
from plan_execute import build_pe_graph
from tools import reset_house


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    col = get_collection()
    print(f"Документів у ChromaDB: {col.count()} (зашито {len(KNOWLEDGE_DOCS)})")
    print(knowledge_search.invoke({"query": "температура взимку перед приїздом"})[:400])
    print("\n[retriever]")
    for h in search_docs("безпека керування пристроєм HITL"):
        print("-", h["id"], h["topic"])

    reset_house()
    app = build_pe_graph()
    # Простий виклик через plan_execute на RAG-запит у складі задачі
    result = app.invoke(
        {
            "task": (
                "Які правила безпеки для remote-керування пристроями "
                "і яка комфортна температура взимку?"
            ),
            "messages": [],
            "plan": [],
            "completed_steps": [],
            "current_step_idx": 0,
            "response": "",
            "replan_count": 0,
        }
    )
    print("\n[RESPONSE]")
    print(result.get("response"))


if __name__ == "__main__":
    main()
