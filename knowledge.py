"""Agentic RAG: база знань Smart Home + knowledge_search (крок 8)."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from langchain_core.tools import tool
from pydantic import BaseModel, Field

CHROMA_DIR = Path(__file__).with_name("chroma_db")
COLLECTION_NAME = "smart_home_knowledge_v1"
SEARCH_K = 3

KNOWLEDGE_DOCS: list[dict[str, str]] = [
    {
        "id": "winter-temp",
        "topic": "temperature",
        "source": "kb_climate",
        "text": (
            "Взимку комфортна температура житлових кімнат — 20–22°C. "
            "Перед приїздом у холод (−10°C і нижче) увімкніть опалення за 1–2 години, "
            "щоб кімнати встигли прогрітися без пікового навантаження."
        ),
    },
    {
        "id": "energy-efficiency",
        "topic": "energy",
        "source": "kb_energy",
        "text": (
            "Енергоефективність: знижуйте target_c на 2–3°C, коли нікого немає вдома. "
            "Увімкнення HVAC і бойлера одночасно збільшує пікове споживання — "
            "краще спочатку HVAC, потім бойлер."
        ),
    },
    {
        "id": "appliance-safety",
        "topic": "safety",
        "source": "kb_safety",
        "text": (
            "Правила безпеки: не залишайте обігрівачі без нагляду на максимумі. "
            "Будь-яке remote-керування пристроєм має проходити через підтвердження людини (HITL). "
            "Замок вхідних дверей не відкривайте автоматично без явного approve."
        ),
    },
    {
        "id": "arrival-scene",
        "topic": "automation",
        "source": "kb_scenes",
        "text": (
            "Сценарій «приїзд додому»: 1) перевірити температуру, 2) увімкнути HVAC, "
            "3) увімкнути світло в коридорі, 4) перевірити розклад автоматизацій. "
            "Керування пристроями — лише після HITL."
        ),
    },
    {
        "id": "hvac-manual",
        "topic": "devices",
        "source": "kb_manual",
        "text": (
            "HVAC living/bedroom: режими heat/cool/off. Рекомендований target узимку 21°C. "
            "device_id: hvac_living, hvac_bedroom. Дії: on, off, set з params.target_c."
        ),
    },
    {
        "id": "lighting-manual",
        "topic": "devices",
        "source": "kb_manual",
        "text": (
            "Освітлення: light_living, light_hallway. Яскравість 0–100. "
            "Для приїзду достатньо hallway brightness 40–60, living — за потреби."
        ),
    },
    {
        "id": "humidity-range",
        "topic": "climate",
        "source": "kb_climate",
        "text": (
            "Оптимальна вологість у житлі 40–60%. Нижче 30% — сухість повітря; "
            "вище 65% — ризик конденсату на холодних стінах узимку."
        ),
    },
    {
        "id": "loop-risk",
        "topic": "safety",
        "source": "kb_safety",
        "text": (
            "Циклічне ввімкнення/вимкнення пристрою шкодить реле і підвищує споживання. "
            "Детекція повторів у агенті зупиняє однакові tool-виклики поспіль."
        ),
    },
    {
        "id": "boiler-tips",
        "topic": "devices",
        "source": "kb_manual",
        "text": (
            "Бойлер (boiler): target_c зазвичай 50–60°C. Не вмикайте разом із усіма HVAC "
            "на повну потужність одразу після приїзду."
        ),
    },
    {
        "id": "schedule-vs-manual",
        "topic": "automation",
        "source": "kb_scenes",
        "text": (
            "Розклад (schedule_list) показує заплановані автоматизації. "
            "Якщо сценарій «приїзд» не покритий розкладом — складіть план вручну "
            "через Plan-and-Execute і підтвердіть device_control."
        ),
    },
]

_collection = None

KEYWORD_TO_DOC: list[tuple[tuple[str, ...], str]] = [
    (("зим", "температур", "прогр", "−10", "-10"), "winter-temp"),
    (("енерг", "спожив", "ефектив"), "energy-efficiency"),
    (("безпек", "hitl", "замок"), "appliance-safety"),
    (("приїзд", "підгот", "сцен"), "arrival-scene"),
    (("hvac", "опален", "клімат"), "hvac-manual"),
    (("світл", "light", "яскрав"), "lighting-manual"),
    (("волог", "humid"), "humidity-range"),
    (("повтор", "цикл", "реле"), "loop-risk"),
    (("бойлер", "boiler"), "boiler-tips"),
    (("розклад", "автоматиз", "schedule"), "schedule-vs-manual"),
]


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    wanted = {d["id"] for d in KNOWLEDGE_DOCS}
    existing = set(collection.get().get("ids") or [])
    if existing != wanted:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        collection.add(
            ids=[d["id"] for d in KNOWLEDGE_DOCS],
            documents=[d["text"] for d in KNOWLEDGE_DOCS],
            metadatas=[{"source": d["source"], "topic": d["topic"]} for d in KNOWLEDGE_DOCS],
        )
    _collection = collection
    return _collection


def _docs_by_ids(collection, ids: list[str]) -> list[dict]:
    if not ids:
        return []
    raw = collection.get(ids=ids)
    hits = []
    for i, doc_id in enumerate(raw.get("ids") or []):
        texts = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        meta = metas[i] if i < len(metas) else {}
        hits.append(
            {
                "id": doc_id,
                "content": texts[i] if i < len(texts) else "",
                "source": meta.get("source", "?"),
                "topic": meta.get("topic", "?"),
            }
        )
    return hits


def search_docs(query: str, k: int = SEARCH_K) -> list[dict]:
    collection = get_collection()
    q = query.casefold()
    boosted_ids = []
    for keys, doc_id in KEYWORD_TO_DOC:
        if any(key in q for key in keys) and doc_id not in boosted_ids:
            boosted_ids.append(doc_id)
    boosted = _docs_by_ids(collection, boosted_ids)
    raw = collection.query(query_texts=[query], n_results=k)
    vector_hits = []
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        vector_hits.append(
            {
                "id": ids[i] if i < len(ids) else "?",
                "content": text,
                "source": meta.get("source", "?"),
                "topic": meta.get("topic", "?"),
            }
        )
    merged, seen = [], set()
    for hit in boosted + vector_hits:
        if hit["id"] in seen:
            continue
        seen.add(hit["id"])
        merged.append(hit)
        if len(merged) >= k:
            break
    return merged


class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., min_length=3, description="Запит до бази знань smart home")


@tool(args_schema=KnowledgeSearchInput)
def knowledge_search(query: str) -> str:
    """Шукає правила smart home: температури, енергія, безпека, сценарії, інструкції.

    Використовуй для рекомендацій і правил. Не використовуй замість sensor_read / device_status.
    """
    hits = search_docs(query)
    if not hits:
        return json.dumps({"status": "not_found", "query": query}, ensure_ascii=False)
    return json.dumps(
        {"status": "ok", "query": query, "results": hits}, ensure_ascii=False
    )
