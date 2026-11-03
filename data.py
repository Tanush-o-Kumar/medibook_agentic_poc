"""Synthetic hospital data: doctors, patients and pre-existing appointments.

All data is fictional and generated relative to today's date so the demo
always has a realistic, partially-booked calendar.
"""
import random
from datetime import datetime, timedelta, date

SLOT_MINUTES = 30
LUNCH_HOUR = 13  # 13:00-14:00 no OPD

# weekday(): Mon=0 ... Sun=6  (clinic closed on Sunday)
DOCTORS = [
    {"id": "D01", "name": "Dr. Ananya Rao",    "specialty": "General Physician", "days": [0, 1, 2, 3, 4, 5], "start": 9,  "end": 17, "room": "OPD-1"},
    {"id": "D02", "name": "Dr. Vikram Shah",   "specialty": "General Physician", "days": [0, 2, 4, 5],       "start": 10, "end": 17, "room": "OPD-2"},
    {"id": "D03", "name": "Dr. Meera Iyer",    "specialty": "Cardiology",        "days": [0, 2, 4],          "start": 9,  "end": 13, "room": "CARD-1"},
    {"id": "D04", "name": "Dr. Arjun Mehta",   "specialty": "Orthopedics",       "days": [1, 3, 5],          "start": 9,  "end": 17, "room": "ORTH-1"},
    {"id": "D05", "name": "Dr. Kavya Nair",    "specialty": "Dermatology",       "days": [0, 1, 3],          "start": 14, "end": 17, "room": "DERM-1"},
    {"id": "D06", "name": "Dr. Rohan Gupta",   "specialty": "Pediatrics",        "days": [0, 1, 2, 3, 4],    "start": 9,  "end": 16, "room": "PED-1"},
    {"id": "D07", "name": "Dr. Sneha Kulkarni","specialty": "ENT",               "days": [1, 2, 4],          "start": 9,  "end": 13, "room": "ENT-1"},
    {"id": "D08", "name": "Dr. Farhan Ali",    "specialty": "Neurology",         "days": [2, 4],             "start": 10, "end": 16, "room": "NEU-1"},
    {"id": "D09", "name": "Dr. Priya Menon",   "specialty": "Gynecology",        "days": [0, 2, 3, 5],       "start": 9,  "end": 15, "room": "GYN-1"},
]

SPECIALTIES = sorted({d["specialty"] for d in DOCTORS})

PATIENTS = [
    {"id": "P001", "name": "Rahul Verma",    "age": 34, "gender": "M", "phone": "+91-90000-00001"},
    {"id": "P002", "name": "Aisha Khan",     "age": 27, "gender": "F", "phone": "+91-90000-00002"},
    {"id": "P003", "name": "Aarav Sharma",   "age": 7,  "gender": "M", "phone": "+91-90000-00003"},
    {"id": "P004", "name": "Lakshmi Pillai", "age": 62, "gender": "F", "phone": "+91-90000-00004"},
    {"id": "P005", "name": "Karan Singh",    "age": 45, "gender": "M", "phone": "+91-90000-00005"},
    {"id": "P006", "name": "Neha Joshi",     "age": 31, "gender": "F", "phone": "+91-90000-00006"},
    {"id": "P007", "name": "Ishaan Das",     "age": 3,  "gender": "M", "phone": "+91-90000-00007"},
    {"id": "P008", "name": "Suresh Reddy",   "age": 58, "gender": "M", "phone": "+91-90000-00008"},
    {"id": "P009", "name": "Fatima Sheikh",  "age": 39, "gender": "F", "phone": "+91-90000-00009"},
    {"id": "P010", "name": "Manoj Patil",    "age": 51, "gender": "M", "phone": "+91-90000-00010"},
]

SEED_REASONS = ["Follow-up", "Routine check-up", "Review reports", "Medication review", "Consultation"]


def doctor_by_id(doc_id):
    return next((d for d in DOCTORS if d["id"] == doc_id), None)


def patient_by_id(pid):
    return next((p for p in PATIENTS if p["id"] == pid), None)


def doctor_slots_on(doc, day: date):
    """All 30-minute slot start times a doctor works on a given day."""
    if day.weekday() not in doc["days"]:
        return []
    slots, t = [], datetime(day.year, day.month, day.day, doc["start"])
    end = datetime(day.year, day.month, day.day, doc["end"])
    while t < end:
        if t.hour != LUNCH_HOUR:
            slots.append(t)
        t += timedelta(minutes=SLOT_MINUTES)
    return slots


def generate_seed_appointments(today: date | None = None, fill_rate=0.45, seed=42):
    """Fill ~45% of the next 10 days' slots with synthetic bookings."""
    rng = random.Random(seed)
    today = today or date.today()
    appts, n = [], 1
    other_patients = [f"P{100 + i}" for i in range(60)]  # anonymous walk-in IDs
    for offset in range(0, 10):
        day = today + timedelta(days=offset)
        for doc in DOCTORS:
            for slot in doctor_slots_on(doc, day):
                if rng.random() < fill_rate:
                    appts.append({
                        "appt_id": f"A{n:04d}",
                        "patient_id": rng.choice(other_patients),
                        "doctor_id": doc["id"],
                        "start": slot.isoformat(timespec="minutes"),
                        "status": "Booked",
                        "reason": rng.choice(SEED_REASONS),
                        "urgency": "Routine",
                        "created_by": "Seed data",
                    })
                    n += 1
    # Give two demo patients an existing future appointment (for cancel / reschedule demos)
    for pid, doc_id, off, hour in [("P005", "D04", 3, 11), ("P006", "D05", 2, 15)]:
        day = today + timedelta(days=off)
        while day.weekday() not in doctor_by_id(doc_id)["days"]:
            day += timedelta(days=1)
        start = datetime(day.year, day.month, day.day, hour).isoformat(timespec="minutes")
        appts = [a for a in appts if not (a["doctor_id"] == doc_id and a["start"] == start)]
        appts.append({"appt_id": f"A{n:04d}", "patient_id": pid, "doctor_id": doc_id, "start": start,
                      "status": "Booked", "reason": "Follow-up", "urgency": "Routine", "created_by": "Seed data"})
        n += 1
    return appts
