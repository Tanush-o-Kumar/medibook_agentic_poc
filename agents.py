"""The agents (LangGraph nodes).

Each agent reads the shared state, does one job, and returns a partial
state update plus a trace entry so the UI can show who did what and why.

  Planner   -> understands the request, extracts details, builds the task plan
  Triage    -> decides specialty + urgency (LLM proposes, safety rules dispose)
  Scheduler -> searches the calendar tool and proposes the best slot
  Critic    -> validates the proposal against hard constraints; can reject
  Executor  -> commits the booking / cancellation to the calendar
  Escalation, Human-review, Notifier -> outcome handling
"""
import time
from datetime import datetime, timedelta

from data import SPECIALTIES, DOCTORS, patient_by_id, doctor_by_id
import tools as T

MAX_ATTEMPTS = 3


def _llm(config):
    return config["configurable"]["llm"]


def _now(config):
    return config["configurable"].get("now") or datetime.now().replace(second=0, microsecond=0)


def _trace(agent, action, detail, t0, source="Rules", status="ok"):
    return {"agent": agent, "action": action, "detail": detail, "source": source,
            "status": status, "ms": int((time.perf_counter() - t0) * 1000)}


# =============================================================== 1. PLANNER
PLANNER_SYS = """You are the Planner agent of a hospital appointment desk.
Read the patient's message and return ONLY JSON with keys:
intent: one of "book", "reschedule", "cancel"
symptoms: short summary of the medical complaint ("" if none)
preferred_date: a date in YYYY-MM-DD if the patient mentioned one, else null
time_preference: one of "morning", "afternoon", "evening", "any"
doctor_name: a doctor's name if mentioned, else null
Today is {today} ({weekday})."""


def planner_agent(state, config):
    t0, llm, now = time.perf_counter(), _llm(config), _now(config)
    text, today = state["request"], _now(config).date()
    patient = patient_by_id(state["patient_id"]) or {"age": None, "name": "Patient"}

    # --- rule extraction (always runs: it is the fallback AND the cross-check)
    low = text.lower()
    rule_intent = "cancel" if any(k in low for k in ["cancel", "call off"]) else \
        "reschedule" if any(k in low for k in ["reschedule", "postpone", "change my appointment", "move my appointment", "prepone"]) else "book"
    rule_date = T.parse_date_pref(text, today)
    rule_time = T.parse_time_pref(text)
    extracted = {"intent": rule_intent, "symptoms": text, "preferred_date": rule_date.isoformat() if rule_date else None,
                 "time_preference": rule_time["label"], "doctor_name": T.find_doctor_mention(text)}
    source = "Rules"

    # --- LLM extraction (if available), validated field-by-field
    out = llm.json(PLANNER_SYS.format(today=today.isoformat(), weekday=today.strftime("%A")), text)
    if out:
        source = f"LLM ({llm.model}) + rule validation"
        if out.get("intent") in ("book", "reschedule", "cancel"):
            extracted["intent"] = out["intent"]
        if out.get("symptoms"):
            extracted["symptoms"] = str(out["symptoms"])
        try:
            d = datetime.fromisoformat(str(out.get("preferred_date"))).date()
            if today <= d <= today + timedelta(days=60):
                extracted["preferred_date"] = d.isoformat()
        except Exception:
            pass
        if rule_time["label"] == "any" and out.get("time_preference") in ("morning", "afternoon", "evening"):
            extracted["time_preference"] = out["time_preference"]
        names = [d["name"] for d in DOCTORS]
        if out.get("doctor_name") in names:
            extracted["doctor_name"] = out["doctor_name"]

    extracted["age"] = patient["age"]
    intent = extracted["intent"]
    plan = {
        "book": ["Extract request details", "Triage: specialty + urgency", "Search calendar for slots",
                 "Critic validates proposed slot", "Book appointment", "Notify patient"],
        "reschedule": ["Extract request details", "Locate existing appointment", "Triage: specialty + urgency",
                       "Search calendar for new slot", "Critic validates", "Cancel old + book new", "Notify patient"],
        "cancel": ["Extract request details", "Locate existing appointment", "Cancel appointment", "Notify patient"],
    }[intent]

    # Reschedule needs an existing appointment; carry its specialty forward
    existing = T.upcoming_for_patient(state["appointments"], state["patient_id"], now)
    detail = {"intent": intent, "extracted": extracted, "plan": plan,
              "existing_appointments": [f'{a["appt_id"]} · {T.doctor_name(a["doctor_id"])} · {a["start"]}' for a in existing]}
    return {"intent": intent, "extracted": extracted, "plan": plan, "existing": existing,
            "trace": [_trace("Planner", f"Intent = {intent}; built {len(plan)}-step plan", detail, t0, source)]}


def route_after_planner(state):
    if state["intent"] == "cancel":
        return "executor"
    if state["intent"] == "reschedule" and not state["existing"]:
        return "human_review"
    return "triage"


# =============================================================== 2. TRIAGE
TRIAGE_SYS = """You are a hospital triage nurse agent. Given symptoms and age, return ONLY JSON:
specialty: exactly one of {specs}
urgency: one of "Emergency", "Urgent", "Routine"
reason: one short sentence.
Emergency = life-threatening now. Urgent = should be seen within 48 hours. Children under 14 go to Pediatrics unless it is a bone injury."""


def triage_agent(state, config):
    t0, llm = time.perf_counter(), _llm(config)
    ex = state["extracted"]
    symptoms = ex["symptoms"]
    if state["intent"] == "reschedule" and state["existing"] and len(symptoms.split()) < 6:
        # carry forward the specialty of the appointment being moved
        old_doc = doctor_by_id(state["existing"][0]["doctor_id"])
        symptoms = f'{symptoms} (existing {old_doc["specialty"]} appointment)'
    rules = T.rule_triage(state["request"] + " " + symptoms, ex["age"])
    if state["intent"] == "reschedule" and state["existing"]:
        rules["specialty"] = doctor_by_id(state["existing"][0]["doctor_id"])["specialty"]
        rules["keywords"] = ["carried over from existing appointment"]

    result = {**rules, "reason": f"Matched keywords {rules['keywords'] or ['(none) -> default GP']}", "llm_agrees": None}
    source = "Rules"
    out = llm.json(TRIAGE_SYS.format(specs=SPECIALTIES), f"Age: {ex['age']}\nSymptoms: {symptoms}")
    if out and out.get("specialty") in SPECIALTIES and out.get("urgency") in ("Emergency", "Urgent", "Routine"):
        source = f"LLM ({llm.model}) + safety guardrails"
        result["llm_agrees"] = out["specialty"] == rules["specialty"]
        if state["intent"] != "reschedule":
            result["specialty"] = out["specialty"]
        result["urgency"] = out["urgency"]
        result["reason"] = out.get("reason", "")
        # GUARDRAILS: red-flag keywords always win; LLM may only raise, never lower, rule urgency
        rank = {"Routine": 0, "Urgent": 1, "Emergency": 2}
        if rank[rules["urgency"]] > rank[result["urgency"]]:
            result["urgency"] = rules["urgency"]
            result["reason"] += f" [Guardrail raised urgency to {rules['urgency']}: {rules['flags']}]"
        if ex["age"] is not None and ex["age"] < T.PEDIATRIC_AGE and result["specialty"] not in ("Pediatrics", "Orthopedics"):
            result["specialty"] = "Pediatrics"
            result["reason"] += " [Guardrail: paediatric patient]"
    return {"triage": result,
            "trace": [_trace("Triage", f'{result["specialty"]} · {result["urgency"]}', result, t0, source)]}


def route_after_triage(state):
    return "escalation" if state["triage"]["urgency"] == "Emergency" else "scheduler"


# =============================================================== 3. SCHEDULER
def _window(state, now):
    urg = state["triage"]["urgency"]
    pref = state["extracted"].get("preferred_date")
    pref_d = datetime.fromisoformat(pref).date() if pref else None
    if urg == "Urgent":           # SLA: within 48 h, preference is secondary
        return now, now + timedelta(hours=48), pref_d
    start = now
    if pref_d and pref_d > now.date():
        start = datetime.combine(pref_d, datetime.min.time())
    return start, start + timedelta(days=7), pref_d


def scheduler_agent(state, config):
    t0, now = time.perf_counter(), _now(config)
    tri, ex = state["triage"], state["extracted"]
    w_start, w_end, pref_d = _window(state, now)
    exclude = {(r["doctor_id"], r["start"]) for r in state.get("rejected", [])}
    cands = T.find_available_slots(state["appointments"], tri["specialty"], max(w_start, now), w_end,
                                   time_pref=T.time_pref_from_label(ex.get("time_preference")), exclude=exclude,
                                   preferred_doctor=ex.get("doctor_name"), patient_id=state["patient_id"],
                                   preferred_date=pref_d)
    # Fallback 1: widen the window by a week if nothing found
    if not cands:
        cands = T.find_available_slots(state["appointments"], tri["specialty"], max(w_start, now), w_end + timedelta(days=7),
                                       exclude=exclude, patient_id=state["patient_id"])
    proposed = cands[0] if cands else None
    appts = state["appointments"]
    trace = [_trace("Scheduler", f"Searched {tri['specialty']} calendar {max(w_start, now):%d %b %H:%M} → {w_end:%d %b}; "
                                 f"{len(cands)} candidates; proposing {proposed['doctor'] + ' @ ' + proposed['start'] if proposed else 'none'}",
                    {"window": [w_start.isoformat(timespec='minutes'), w_end.isoformat(timespec='minutes')],
                     "excluded_after_critic_rejection": sorted(exclude), "top_candidates": cands}, t0, "Tool: find_available_slots")]

    # Demo: simulate a race condition - front desk books the same slot a moment later
    if proposed and config["configurable"].get("simulate_conflict") and state.get("attempts", 0) == 0:
        appts, _ = T.book_slot(appts, "P999", proposed["doctor_id"], proposed["start"], "Walk-in (front desk)", "Routine")
        trace.append(_trace("External event", "Front-desk staff booked the same slot concurrently (simulated)",
                            {"slot": proposed}, time.perf_counter(), "Simulation", "warn"))
    return {"candidates": cands, "proposed": proposed, "appointments": appts, "trace": trace}


# =============================================================== 4. CRITIC
def critic_agent(state, config):
    t0, now = time.perf_counter(), _now(config)
    p, tri, ex = state["proposed"], state["triage"], state["extracted"]
    attempts = state.get("attempts", 0) + 1
    if not p:
        checks = [{"check": "A slot was proposed", "passed": False, "hard": True, "note": "No availability found"}]
        return {"critic": {"verdict": "NO_SLOT", "checks": checks}, "attempts": MAX_ATTEMPTS,
                "trace": [_trace("Critic", "No slot to validate → human review", checks, t0, status="fail")]}
    doc, start = doctor_by_id(p["doctor_id"]), datetime.fromisoformat(p["start"])
    from data import doctor_slots_on
    sla_h = 48 if tri["urgency"] == "Urgent" else 24 * 15
    checks = [
        {"check": "Doctor specialty matches triage", "passed": doc["specialty"] == tri["specialty"], "hard": True,
         "note": f'{doc["specialty"]} vs {tri["specialty"]}'},
        {"check": "Slot is in the future", "passed": start > now, "hard": True, "note": p["start"]},
        {"check": "Within doctor's working hours", "passed": start in doctor_slots_on(doc, start.date()), "hard": True,
         "note": f'{doc["name"]} works {doc["start"]}:00–{doc["end"]}:00'},
        {"check": "Doctor not double-booked (live ledger)", "passed": not T.doctor_busy(state["appointments"], doc["id"], p["start"]),
         "hard": True, "note": "re-checked against latest calendar"},
        {"check": "Patient has no clashing appointment", "passed": not T.patient_busy(state["appointments"], state["patient_id"], p["start"]),
         "hard": True, "note": state["patient_id"]},
        {"check": f"Urgency SLA met (≤{sla_h} h)", "passed": (start - now) <= timedelta(hours=sla_h), "hard": True,
         "note": f"{(start - now).total_seconds() / 3600:.1f} h from now"},
        {"check": "Time-of-day preference respected", "passed": p["matches_time_pref"], "hard": False,
         "note": ex.get("time_preference")},
        {"check": "Preferred date respected", "passed": (not ex.get("preferred_date")) or p["start"].startswith(ex["preferred_date"]),
         "hard": False, "note": ex.get("preferred_date") or "none given"},
        {"check": "LLM triage agrees with symptom knowledge base", "passed": tri.get("llm_agrees") is not False, "hard": False,
         "note": "rules-only run" if tri.get("llm_agrees") is None else ("agree" if tri["llm_agrees"] else "disagree → flag for clinician review")},
    ]
    hard_fail = [c for c in checks if c["hard"] and not c["passed"]]
    verdict = "APPROVE" if not hard_fail else "REJECT"
    upd = {"critic": {"verdict": verdict, "checks": checks}, "attempts": attempts}
    if hard_fail:
        upd["rejected"] = state.get("rejected", []) + [{"doctor_id": p["doctor_id"], "start": p["start"]}]
    msg = f"{verdict} (attempt {attempts}/{MAX_ATTEMPTS})" + (f": failed {[c['check'] for c in hard_fail]}" if hard_fail else
                                                             f"; {sum(c['passed'] for c in checks)}/{len(checks)} checks passed")
    return {**upd, "trace": [_trace("Critic", msg, checks, t0, "Rules (constraint validator)", "ok" if not hard_fail else "fail")]}


def route_after_critic(state):
    v = state["critic"]["verdict"]
    if v == "APPROVE":
        return "executor"
    return "scheduler" if state["attempts"] < MAX_ATTEMPTS else "human_review"


# =============================================================== 5. EXECUTOR
def executor_agent(state, config):
    t0 = time.perf_counter()
    appts, intent = state["appointments"], state["intent"]
    actions, booking, cancelled = [], None, None
    if intent in ("cancel", "reschedule"):
        if not state["existing"]:
            return {"outcome": "NOT_FOUND",
                    "trace": [_trace("Executor", "No upcoming appointment to cancel", {}, t0, "Tool: cancel_appt", "warn")]}
        cancelled = state["existing"][0]
        appts = T.cancel_appt(appts, cancelled["appt_id"])
        actions.append(f'cancel_appt({cancelled["appt_id"]})')
    if intent in ("book", "reschedule"):
        p = state["proposed"]
        appts, booking = T.book_slot(appts, state["patient_id"], p["doctor_id"], p["start"],
                                     state["extracted"]["symptoms"], state["triage"]["urgency"])
        actions.append(f'book_slot({p["doctor_id"]}, {p["start"]}) → {booking["appt_id"]}')
        # post-commit verification
        dup = sum(1 for a in appts if a["status"] == "Booked" and a["doctor_id"] == p["doctor_id"] and a["start"] == p["start"])
        actions.append(f"post-commit check: {dup} booking(s) in slot → {'OK' if dup == 1 else 'CONFLICT'}")
    outcome = {"book": "BOOKED", "reschedule": "RESCHEDULED", "cancel": "CANCELLED"}[intent]
    return {"appointments": appts, "booking": booking, "cancelled": cancelled, "outcome": outcome,
            "trace": [_trace("Executor", f"{outcome}: " + "; ".join(actions), {"booking": booking, "cancelled": cancelled}, t0,
                             "Tools: book_slot / cancel_appt")]}


# =============================================================== 6. OUTCOME HANDLERS
def escalation_agent(state, config):
    t0 = time.perf_counter()
    alert = {"alert": "EMERGENCY", "patient": state["patient_id"], "flags": state["triage"]["flags"],
             "action": "Paged on-duty ER doctor; no OPD slot booked"}
    return {"outcome": "ESCALATED", "booking": None,
            "trace": [_trace("Escalation", "Red-flag symptoms → ER alert raised, booking skipped", alert, t0, "Safety policy", "warn")]}


def human_review_agent(state, config):
    t0 = time.perf_counter()
    reason = "No existing appointment to reschedule" if state["intent"] == "reschedule" and not state.get("existing") \
        else f"Critic rejected {state.get('attempts', 0)} proposal(s) / no slot available"
    return {"outcome": "NEEDS_HUMAN", "booking": None,
            "trace": [_trace("Human-in-loop", f"Added to front-desk review queue: {reason}", {"reason": reason}, t0, "Policy", "warn")]}


NOTIFY_SYS = """You write short, warm SMS messages from 'City Care Hospital' to patients.
Use ONLY the facts given. Max 60 words. No medical advice beyond what is given."""


def notifier_agent(state, config):
    t0, llm = time.perf_counter(), _llm(config)
    pat = patient_by_id(state["patient_id"]) or {"name": "Patient"}
    first = pat["name"].split()[0]
    outcome, b, c = state.get("outcome"), state.get("booking"), state.get("cancelled")
    must_include = []
    if outcome in ("BOOKED", "RESCHEDULED"):
        doc = doctor_by_id(b["doctor_id"])
        when = T.pretty_dt(b["start"])
        facts = f"Outcome: {outcome}. Patient: {first}. Doctor: {doc['name']} ({doc['specialty']}). When: {when}. Room: {doc['room']}. Ref: {b['appt_id']}. Arrive 15 minutes early."
        if c:
            facts += f" Previous appointment {c['appt_id']} on {T.pretty_dt(c['start'])} is cancelled."
        template = (f"Hi {first}, your appointment with {doc['name']} ({doc['specialty']}) is confirmed for {when}, "
                    f"Room {doc['room']}. Ref {b['appt_id']}. Please arrive 15 min early. Reply C to cancel. – City Care Hospital")
        if c:
            template = f"Hi {first}, your appointment {c['appt_id']} has been moved. " + template.split(", ", 1)[1]
        must_include = [doc["name"].split()[-1], b["appt_id"]]
    elif outcome == "CANCELLED":
        facts = f"Outcome: cancelled. Patient: {first}. Appointment {c['appt_id']} with {T.doctor_name(c['doctor_id'])} on {T.pretty_dt(c['start'])}."
        template = f"Hi {first}, your appointment {c['appt_id']} with {T.doctor_name(c['doctor_id'])} on {T.pretty_dt(c['start'])} has been cancelled. Reply BOOK to schedule again. – City Care Hospital"
        must_include = [c["appt_id"]]
    elif outcome == "ESCALATED":
        facts = template = (f"URGENT – {first}, your symptoms may need emergency care. Please go to the nearest emergency room "
                            f"or call 108 immediately. Our ER team has been alerted. – City Care Hospital")
        return {"message": template, "trace": [_trace("Notifier", "Fixed safety template used (LLM not allowed for emergencies)",
                                                      {"message": template}, t0, "Template")]}
    elif outcome == "NOT_FOUND":
        facts = template = f"Hi {first}, we couldn't find an upcoming appointment under your ID. Our front desk will call you shortly. – City Care Hospital"
    else:
        facts = template = f"Hi {first}, we've received your request. Our front desk will call you within 2 hours to confirm a slot. – City Care Hospital"

    msg, source = template, "Template"
    draft = llm.text(NOTIFY_SYS, facts) if must_include else None
    if draft:
        # mini-critic on the LLM's draft: must contain key facts, else fall back
        if all(k in draft for k in must_include) and len(draft) < 600:
            msg, source = draft, f"LLM ({llm.model}), fact-checked"
        else:
            source = "Template (LLM draft failed fact-check)"
    return {"message": msg, "trace": [_trace("Notifier", "Drafted patient SMS", {"message": msg, "required_facts": must_include}, t0, source)]}
