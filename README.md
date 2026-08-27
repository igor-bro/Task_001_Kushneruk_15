# Task_001 — Варіант 15: Smart Home «Home Assistant»

Агент розумного дому: сенсори, статуси, енергія, розклади, HITL-керування,
ReAct + Plan-and-Execute, SqliteSaver, Agentic RAG (ChromaDB).

Репо: https://github.com/igor-bro/Task_001_Kushneruk_15

## Статус

- [x] Крок 1: середовище
- [x] Крок 2: tools + Pydantic
- [x] Крок 3–5: ReAct + max_steps/timeout/loop + trajectory
- [x] Крок 6: Plan-and-Execute
- [x] Крок 7: SqliteSaver
- [x] Крок 8: Agentic RAG
- [x] Крок 9: HITL (`device_control`)
- [x] Крок 10: pytest + README

## Швидкий старт

```powershell
cd Task_001_Kushneruk_15
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy ..\hw2_kushneruk_agent\.env .env
python check_setup.py
python demo_tools.py
pytest test_tools.py -v
python demo_safety.py
python react_agent.py
python plan_execute.py
python checkpointer_demo.py
python demo_rag.py
python hitl.py start
python hitl.py approve
# або: python hitl.py start && python hitl.py reject
```

## Архітектура

```text
ReAct:     START → agent ⇄ tools → END
P&E:       START → planner → executor(вкладений ReAct) → replanner ⇄ executor | END
HITL:      agent → safe_tools | risky_tools(interrupt_before) → agent
```

### Tools

| Tool | Тип | JSON-вихід |
| --- | --- | --- |
| `sensor_read` | safe | `{status, data}` |
| `device_status` | safe | `{status, data}` |
| `energy_consumption` | safe | `{status, data}` |
| `schedule_list` | safe | `{status, data}` |
| `knowledge_search` | safe | RAG |
| `device_control` | **risky** | HITL |

### P&E сценарій

«Я повернуся через 2 години. Зима, −10°C. Підготуй дім» —
читає сенсори/статуси, шукає правила в RAG, пропонує план (керування лише через HITL).

## Аналітичні питання

1. **Чому будь-яке керування пристроєм потребує HITL?**  
   Side effect незворотний у реальному часі (світло, тепло, замок). LLM може
   помилитись у `device_id`/`action`; approve дає людині контроль перед зміною стану.

2. **Як Plan-and-Execute декомпозує підготовку дому?**  
   Planner розбиває на перевірку температури, статусів HVAC, розкладу й пошук
   рекомендацій; executor виконує по одному кроку; replanner збирає фінальний план дій.

3. **Як checkpointer зберігає стан між сесіями?**  
   `SqliteSaver` пише checkpoint після кожного вузла за `thread_id`. Новий процес
   з тим самим thread відновлює історію повідомлень через `get_state()` / `invoke`.

4. **Навіщо RAG із правилами безпеки, якщо є `device_control`?**  
   Tool змінює стан; RAG дає політику (коли/як безпечно керувати). Без RAG агент
   може вмикати все одразу або відкрити замок без контексту правил.

5. **Як детекція повторів захищає від циклічного on/off?**  
   `LoopDetector` бачить N однакових `tool+args` поспіль і зупиняє цикл, щоб не
   смикати реле й не палити енергію.

## Структура

```
tools.py, knowledge.py, models.py, safety.py, logger.py
react_agent.py, plan_execute.py
checkpointer_demo.py, hitl.py, demo_*.py
test_tools.py, trajectory.json, requirements.txt, README.md
```
