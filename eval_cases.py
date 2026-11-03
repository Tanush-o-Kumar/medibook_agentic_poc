"""Labelled synthetic test cases used by the Evaluation tab to measure the agents."""

SAMPLE_REQUESTS = {
    "Fever & cough (routine GP)": ("P001", "I've had a mild fever and cough since yesterday, can I see a doctor tomorrow morning?"),
    "Child with ear pain": ("P003", "My son has ear pain and a runny nose, please book something this week in the afternoon"),
    "Knee injury – urgent": ("P010", "I twisted my knee playing cricket, it is swollen and very painful, can't walk properly"),
    "Chest pain – EMERGENCY": ("P004", "My mother has severe chest pain with sweating and difficulty breathing right now"),
    "Skin rash with doctor preference": ("P002", "I have an itchy skin rash on my arms, I'd like to see Dr. Nair on Monday after 3 pm"),
    "Migraine follow-up": ("P008", "Recurring migraine headaches for two weeks, need a neurologist next week"),
    "Reschedule existing": ("P005", "Please reschedule my ortho appointment to an afternoon slot"),
    "Cancel existing": ("P006", "I want to cancel my appointment, I'm travelling"),
    "Pregnancy check-up": ("P009", "I'm 12 weeks pregnant and need a routine check-up on Wednesday morning"),
    "BP monitoring": ("P010", "My BP readings have been high for a week, want to see a heart specialist"),
}

# (patient_id, request, expected_intent, expected_specialty or None, expected_urgency or None, expected_outcome)
EVAL_CASES = [
    ("P001", "Mild fever and cough since yesterday, tomorrow morning please", "book", "General Physician", "Routine", "BOOKED"),
    ("P003", "My son has ear pain and runny nose", "book", "Pediatrics", "Routine", "BOOKED"),
    ("P010", "Twisted knee, swollen and very painful, can't walk", "book", "Orthopedics", "Urgent", "BOOKED"),
    ("P004", "Severe chest pain with sweating and difficulty breathing", "book", "Cardiology", "Emergency", "ESCALATED"),
    ("P002", "Itchy skin rash on arms, Dr. Nair on Monday after 3 pm", "book", "Dermatology", "Routine", "BOOKED"),
    ("P008", "Recurring migraine headaches for two weeks", "book", "Neurology", "Routine", "BOOKED"),
    ("P009", "12 weeks pregnant, routine check-up Wednesday morning", "book", "Gynecology", "Routine", "BOOKED"),
    ("P010", "BP readings high for a week, need heart specialist", "book", "Cardiology", "Routine", "BOOKED"),
    ("P001", "Sore throat and blocked sinus for 4 days", "book", "ENT", "Routine", "BOOKED"),
    ("P007", "Baby has high fever and vomiting since morning", "book", "Pediatrics", "Urgent", "BOOKED"),
    ("P006", "Please cancel my appointment", "cancel", None, None, "CANCELLED"),
    ("P005", "Reschedule my appointment to an afternoon slot", "reschedule", "Orthopedics", None, "RESCHEDULED"),
    ("P002", "Please reschedule my appointment to Friday", "reschedule", None, None, "NEEDS_HUMAN"),
    ("P008", "My father fainted and has slurred speech", "book", None, "Emergency", "ESCALATED"),
    ("P009", "Acne and hair fall problem, any day next week", "book", "Dermatology", "Routine", "BOOKED"),
]
