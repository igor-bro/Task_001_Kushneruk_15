"""Перевірка середовища Task_001 (варіант 15)."""

from __future__ import annotations

import os
import sys


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1) Імпорти ===")
    import chromadb  # noqa: F401
    import langchain_mistralai  # noqa: F401
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    from langgraph.types import Command, interrupt  # noqa: F401

    print("langgraph / mistral / chromadb / interrupt: OK")

    print("\n=== 2) Mistral ===")
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-"):
        print("SKIP — немає MISTRAL_API_KEY у .env")
        return 0

    from langchain_mistralai import ChatMistralAI

    model = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    llm = ChatMistralAI(model=model, temperature=0.1)
    response = llm.invoke("Say OK in one word")
    print(f"Mistral ({model}): {response.content}")
    print("\nПідготовка: УСЕ OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
