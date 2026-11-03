"""Deterministic tools the agents call (the 'hands' of the system).

Agents decide WHAT to do; these functions DO it reliably: parse dates,
look up the symptom knowledge base, search the calendar, book and cancel.
"""
import re
from datetime import datetime, timedelta, date

from data import DOCTORS, doctor_slots_on, doctor_by_id

# ---------------------------------------------------------------- knowledge base
# (keywords, specialty) - checked in order, first match wins
SYMPTOM_KB = [
    (["chest pain", "palpitation", "heart", "bp", "blood pressure", "hypertension", "ecg"], "Cardiology"),
    (["fracture", "bone", "joint", "knee", "back pain", "shoulder", "sprain", "ankle", "spine", "neck pain"], "Orthopedics"),
    (["rash", "skin", "acne", "itch", "eczema", "pimple", "hair fall", "allergy on skin"], "Dermatology"),
    (["ear", "nose", "throat", "sinus", "tonsil", "hearing", "nosebleed", "hoarse"], "ENT"),
    (["migraine", "headache", "seizure", "numbness", "dizziness", "memory", "tremor", "fits"], "Neurology"),
    (["pregnan", "period", "menstrua", "pcos", "pelvic", "gynae", "gyne"], "Gynecology"),
    (["vaccin", "child", "baby", "infant", "kid"], "Pediatrics"),
    (["fever", "cold", "cough", "flu", "stomach", "vomit", "diarrh", "fatigue", "weakness", "check-up",
      "checkup", "diabetes", "sugar", "body pain", "infection"], "General Physician"),
]

# Guardrail keyword lists - never overridden by the LLM
RED_FLAGS = ["chest pain with", "severe chest pain", "crushing chest", "can't breathe", "cannot breathe",
             "difficulty breathing", "unconscious", "fainted", "stroke", "slurred speech", "face drooping",
             "heavy bleeding", "severe bleeding", "suicid", "poison", "overdose", "seizure now", "not breathing"]
URGENT_FLAGS = ["high fever", "severe", "fracture", "since 3 days", "for 3 days", "vomiting", "blood",
                "can't walk", "cannot walk", "swollen", "worsening", "urgent", "asap", "very painful"]

PEDIATRIC_AGE = 14


def rule_triage(text: str, age: int | None):
    t = text.lower()
    hits = [kw for kw in RED_FLAGS if kw in t]
    if hits:
        urgency = "Emergency"
    elif any(kw in t for kw in URGENT_FLAGS):
        urgency, hits = "Urgent", [kw for kw in URGENT_FLAGS if kw in t]
    else:
        urgency = "Routine"
    specialty, matched = "General Physician", []
    for kws, spec in SYMPTOM_KB:
        m = [k for k in kws if k in t]
        if m:
            specialty, matched = spec, m
            break
    # Children go to pediatrics unless it's clearly a surgical/bone issue
    if age is not None and age < PEDIATRIC_AGE and specialty not in ("Orthopedics",):
        specialty = "Pediatrics"
        matched = matched + [f"age {age} < {PEDIATRIC_AGE}"]
    return {"specialty": specialty, "urgency": urgency, "keywords": matched, "flags": hits}


# ---------------------------------------------------------------- parsing
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_date_pref(text: str, today: date):
    t = text.lower()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        try:
            return date(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})[/.](\d{1,2})\b", t)  # dd/mm
    if m:
        try:
            d = date(today.year, int(m[2]), int(m[1]))
            return d if d >= today else date(today.year + 1, d.month, d.day)
        except ValueError:
            pass
    if "day after tomorrow" in t:
        return today + timedelta(days=2)
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if "today" in t or "asap" in t:
        return today
    for i, wd in enumerate(WEEKDAYS):
        if wd in t:
            # "monday" / "next monday" both mean the coming monday (never today)
            delta = (i - today.weekday()) % 7 or 7
            return today + timedelta(days=delta)
    if "next week" in t:
        return today + timedelta(days=(7 - today.weekday()))
    return None


def parse_time_pref(text: str):
    t = text.lower()
    m = re.search(r"after (\d{1,2})\s*(am|pm)?", t)
    if m:
        h = int(m[1]) + (12 if m[2] == "pm" and int(m[1]) < 12 else 0)
        return {"label": f"after {h}:00", "from": h, "to": 24}
    m = re.search(r"before (\d{1,2})\s*(am|pm)?", t)
    if m:
        h = int(m[1]) + (12 if m[2] == "pm" and int(m[1]) < 12 else 0)
        return {"label": f"before {h}:00", "from": 0, "to": h}
    if "morning" in t:
        return {"label": "morning", "from": 0, "to": 12}
    if "afternoon" in t:
        return {"label": "afternoon", "from": 12, "to": 17}
    if "evening" in t:
        return {"label": "evening", "from": 15, "to": 24}
    return {"label": "any", "from": 0, "to": 24}


def time_pref_from_label(label: str | None):
    return parse_time_pref(label or "")


def find_doctor_mention(text: str):
    t = text.lower()
    for d in DOCTORS:
        surname = d["name"].split()[-1].lower()
        if f"dr. {surname}" in t or f"dr {surname}" in t or d["name"].lower() in t:
            return d["name"]
    return None


# ---------------------------------------------------------------- calendar tools
def _active(appts):
    return [a for a in appts if a["status"] == "Booked"]


def doctor_busy(appts, doc_id, start_iso):
    return any(a["doctor_id"] == doc_id and a["start"] == start_iso for a in _active(appts))


def patient_busy(appts, pid, start_iso):
    return any(a["patient_id"] == pid and a["start"] == start_iso for a in _active(appts))


def doctor_load(appts, doc_id, day_iso):
    return sum(1 for a in _active(appts) if a["doctor_id"] == doc_id and a["start"].startswith(day_iso))


def find_available_slots(appts, specialty, window_start: datetime, window_end: datetime,
                         time_pref=None, exclude=(), preferred_doctor=None, patient_id=None,
                         preferred_date: date | None = None, limit=5):
    """Search the live calendar and return ranked free slots."""
    time_pref = time_pref or {"from": 0, "to": 24}
    docs = [d for d in DOCTORS if d["specialty"] == specialty]
    cands = []
    day = window_start.date()
    while day <= window_end.date():
        for doc in docs:
            for slot in doctor_slots_on(doc, day):
                iso = slot.isoformat(timespec="minutes")
                if slot <= window_start or slot > window_end or (doc["id"], iso) in exclude:
                    continue
                if doctor_busy(appts, doc["id"], iso) or (patient_id and patient_busy(appts, patient_id, iso)):
                    continue
                in_pref = time_pref["from"] <= slot.hour < time_pref["to"]
                score = 0.0
                score += 0 if in_pref else 50                                   # respect time preference
                score += 0 if (preferred_date is None or slot.date() == preferred_date) else \
                    10 * abs((slot.date() - preferred_date).days)                # respect date preference
                score += 0 if (preferred_doctor is None or doc["name"] == preferred_doctor) else 30
                score += (slot - window_start).total_seconds() / 3600 * 0.2      # earlier is better
                score += doctor_load(appts, doc["id"], slot.date().isoformat()) * 0.5  # load-balance doctors
                cands.append({"doctor_id": doc["id"], "doctor": doc["name"], "specialty": specialty,
                              "room": doc["room"], "start": iso, "score": round(score, 2),
                              "matches_time_pref": in_pref})
        day += timedelta(days=1)
    cands.sort(key=lambda c: c["score"])
    return cands[:limit]


def book_slot(appts, patient_id, doctor_id, start_iso, reason, urgency):
    new_id = f"A{len(appts) + 1:04d}"
    while any(a["appt_id"] == new_id for a in appts):
        new_id = f"A{int(new_id[1:]) + 1:04d}"
    appt = {"appt_id": new_id, "patient_id": patient_id, "doctor_id": doctor_id, "start": start_iso,
            "status": "Booked", "reason": reason[:80], "urgency": urgency, "created_by": "MediBook Agents"}
    return appts + [appt], appt


def upcoming_for_patient(appts, patient_id, now: datetime):
    ups = [a for a in _active(appts) if a["patient_id"] == patient_id and datetime.fromisoformat(a["start"]) > now]
    return sorted(ups, key=lambda a: a["start"])


def cancel_appt(appts, appt_id):
    out = []
    for a in appts:
        out.append({**a, "status": "Cancelled"} if a["appt_id"] == appt_id else a)
    return out


def pretty_dt(iso: str):
    dt = datetime.fromisoformat(iso)
    return dt.strftime("%a, %d %b %Y at %I:%M %p")


def doctor_name(doc_id):
    d = doctor_by_id(doc_id)
    return d["name"] if d else doc_id
