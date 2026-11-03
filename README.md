# MediBook Agents — Agentic AI PoC for Hospital Appointment Scheduling

**Domain:** Healthcare · **Stack:** Python, Streamlit, LangGraph, Ollama (free local LLM) · **Data:** 100% synthetic

## Problem
Hospital OPD front desks handle appointment requests by phone/WhatsApp manually: listen to symptoms, guess the
right department, check several doctors' calendars, avoid clashes, book, and send a confirmation. It takes ~5–10 min
per request, causes wrong-department bookings and double bookings, and urgent cases can wait in the same queue as routine ones.

## Solution — a team of agents (LangGraph StateGraph)
| Agent | Job |
|---|---|
| Planner / Intake | Understands free text → intent (book / reschedule / cancel), date, time, doctor; builds the task plan |
| Triage | Symptoms + age → specialty + urgency. LLM proposes, **safety guardrails** (red-flag keywords) can only raise urgency |
| Scheduler | Calls `find_available_slots` tool, ranks slots by preference, earliness, doctor load |
| Critic | 6 hard + 3 soft checks on the **live** calendar. Rejects → Scheduler retries (max 3) → else human review |
| Executor | `book_slot` / `cancel_appt` + post-commit duplicate check |
| Escalation | Emergency → ER alert, no OPD booking |
| Human review | Fallback queue for the front desk |
| Notifier | Drafts patient SMS; LLM draft is fact-checked, falls back to template |

## Run it (Mac)
```bash
# 1. Python deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Free local LLM (optional — app falls back to rules if Ollama is off)
brew install ollama        # or download from ollama.com
ollama serve &             # (the Ollama desktop app also starts this)
ollama pull llama3.2       # ~2 GB, 3B model; use qwen2.5:1.5b for a faster laptop demo

# 3. Launch
streamlit run app.py
```
Sidebar → **Agent brain**: *Auto* (Ollama if running, else rules), *Ollama*, or *Rules only*.

## 5-minute demo flow
1. **Fever & cough** sample → show trace: Planner → Triage → Scheduler → Critic (approve) → Executor → Notifier.
2. Turn on **Simulate concurrent booking conflict**, run **Knee injury – urgent** → Critic catches the double booking, Scheduler retries (agent loop).
3. **Chest pain – EMERGENCY** → guardrail escalates to ER; no booking.
4. **Reschedule existing** / **Cancel existing** → Planner routes to a different path.
5. **Evaluation** tab → run 15 labelled cases: accuracy, 0 double-bookings, time per request, business-impact estimate.
6. **Architecture** tab → diagram + agent responsibilities table + Mermaid generated from the compiled LangGraph.

## Files
`app.py` UI · `graph.py` LangGraph wiring · `agents.py` agent logic · `tools.py` calendar tools & knowledge base ·
`llm.py` Ollama client with fallback · `data.py` synthetic doctors/patients/appointments · `eval_cases.py` test set

## Limitations
Keyword KB is small; triage is **not** clinical diagnosis; in-memory calendar (no DB); single clinic; LLM output quality depends on the local model; no real SMS/EHR integration.

## Future scope
Real EHR/HMS integration (FHIR), WhatsApp/SMS channel, voice intake, waitlist auto-fill on cancellations, no-show prediction,
multilingual intake (Hindi/regional), MCP server exposing calendar tools, clinician-approved triage protocols.
