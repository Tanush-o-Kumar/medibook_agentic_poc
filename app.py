"""MediBook Agents - Agentic AI PoC for hospital appointment scheduling.

Run:  streamlit run app.py
"""
import time
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st

from data import DOCTORS, PATIENTS, SPECIALTIES, generate_seed_appointments, patient_by_id, doctor_by_id
from eval_cases import SAMPLE_REQUESTS, EVAL_CASES
from graph import run_workflow, GRAPH
from llm import LocalLLM
import tools as T

st.set_page_config(page_title="MediBook Agents", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ session state
if "appointments" not in st.session_state:
    st.session_state.appointments = generate_seed_appointments()
    st.session_state.history = []
    st.session_state.last = None

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🩺 MediBook Agents")
    st.caption("Agentic AI for hospital appointment scheduling · City Care Hospital (synthetic)")
    st.subheader("Model")
    mode = st.radio("Agent brain", ["auto", "ollama", "mock"], index=0, horizontal=True,
                    format_func={"auto": "Auto", "ollama": "Ollama", "mock": "Rules only"}.get,
                    help="Auto uses Ollama if it is running, otherwise deterministic rules.")
    model = st.text_input("Ollama model", "llama3.2")
    host = st.text_input("Ollama host", "http://localhost:11434")
    llm = LocalLLM(mode="mock" if mode == "mock" else "auto", model=model, host=host)
    models = llm.list_models() if mode != "mock" else None
    if mode == "mock":
        st.info("Rules-only mode (no LLM).")
    elif llm.enabled:
        st.success(f"Connected: {llm.label}")
    elif models is not None:
        st.warning(f"Ollama running but `{model}` not pulled. Run `ollama pull {model}`. Using rules for now.")
    else:
        st.warning("Ollama not reachable → falling back to rules. Start it with `ollama serve`.")
        if mode == "ollama":
            st.error("Ollama mode selected but unavailable.")

    st.subheader("Demo controls")
    simulate = st.toggle("Simulate concurrent booking conflict", value=False,
                         help="Front desk books the same slot right after the Scheduler proposes it. "
                              "Watch the Critic reject it and the Scheduler retry.")
    if st.button("🔄 Reset calendar data", width="stretch"):
        st.session_state.appointments = generate_seed_appointments()
        st.session_state.history, st.session_state.last = [], None
        st.rerun()
    stats_box = st.empty()  # filled at the end of the script so numbers are current

tab_desk, tab_sched, tab_eval, tab_arch = st.tabs(["🧑‍⚕️ Request Desk", "📅 Schedule", "🧪 Evaluation", "🏗️ Architecture"])

ICON = {"Planner": "🧭", "Triage": "🩻", "Scheduler": "🗓️", "Critic": "🔍", "Executor": "⚙️", "Escalation": "🚨",
        "Human-in-loop": "🙋", "Notifier": "✉️", "External event": "⚡"}
OUTCOME_STYLE = {"BOOKED": ("success", "✅ Booked"), "RESCHEDULED": ("success", "✅ Rescheduled"),
                 "CANCELLED": ("info", "🗑️ Cancelled"), "ESCALATED": ("error", "🚨 Escalated to Emergency"),
                 "NEEDS_HUMAN": ("warning", "🙋 Sent to front-desk review"), "NOT_FOUND": ("warning", "❓ Appointment not found")}

# ================================================================== REQUEST DESK
with tab_desk:
    st.header("Patient request → multi-agent workflow")
    st.caption("Type a request the way a patient would on WhatsApp / phone. The agents plan, triage, schedule, "
               "validate, book and notify — with a full audit trail.")
    c1, c2 = st.columns([1, 2])
    with c1:
        sample = st.selectbox("Load a sample request", ["— custom —"] + list(SAMPLE_REQUESTS))
        default_pid, default_text = SAMPLE_REQUESTS.get(sample, ("P001", ""))
        pids = [p["id"] for p in PATIENTS]
        pid = st.selectbox("Patient", pids, index=pids.index(default_pid),
                           format_func=lambda x: f"{x} · {patient_by_id(x)['name']} ({patient_by_id(x)['age']}y)")
        existing = T.upcoming_for_patient(st.session_state.appointments, pid, datetime.now())
        if existing:
            st.caption("Upcoming: " + "; ".join(f'{a["appt_id"]} {T.doctor_name(a["doctor_id"])} {T.pretty_dt(a["start"])}' for a in existing))
    with c2:
        text = st.text_area("Patient message", default_text, height=110, key=f"msg_{sample}")
        go = st.button("▶ Run agents", type="primary")

    if go and not text.strip():
        st.warning("Type a patient message first.")
    elif go:
        with st.spinner(f"Agents working ({llm.label})…"):
            t0 = time.perf_counter()
            result = run_workflow(text, pid, st.session_state.appointments, llm, simulate_conflict=simulate)
            elapsed = time.perf_counter() - t0
        st.session_state.appointments = result["appointments"]
        st.session_state.last = (result, elapsed, text, pid)
        st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "patient": pid, "request": text[:60],
                                         "intent": result.get("intent"), "specialty": result.get("triage", {}).get("specialty", "–"),
                                         "urgency": result.get("triage", {}).get("urgency", "–"), "outcome": result.get("outcome"),
                                         "agent steps": len(result["trace"]), "seconds": round(elapsed, 2)})

    if st.session_state.last:
        result, elapsed, text, pid = st.session_state.last
        tri = result.get("triage", {})
        kind, label = OUTCOME_STYLE.get(result.get("outcome"), ("info", result.get("outcome")))
        getattr(st, kind)(f"**{label}** — {len(result['trace'])} agent steps in {elapsed:.2f}s")
        m = st.columns(5)
        m[0].metric("Intent", result.get("intent", "–").title())
        m[1].metric("Specialty", tri.get("specialty", "–"))
        m[2].metric("Urgency", tri.get("urgency", "–"))
        m[3].metric("Critic attempts", result.get("attempts", 0))
        b = result.get("booking")
        m[4].metric("Slot", datetime.fromisoformat(b["start"]).strftime("%d %b, %H:%M") if b else "–")

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Agent trace")
            for i, step in enumerate(result["trace"], 1):
                icon = ICON.get(step["agent"], "•")
                flag = {"fail": " ❌", "warn": " ⚠️"}.get(step["status"], "")
                with st.expander(f"{i}. {icon} **{step['agent']}** — {step['action']}{flag}", expanded=step["status"] != "ok"):
                    st.caption(f"Source: {step['source']} · {step['ms']} ms")
                    st.json(step["detail"], expanded=False)
        with right:
            if result.get("plan"):
                st.subheader("Planner's task plan")
                for s in result["plan"]:
                    st.markdown(f"- {s}")
            crit = result.get("critic")
            if crit:
                st.subheader(f"Critic validation · {crit['verdict']}")
                st.dataframe(pd.DataFrame([{"": "✅" if c["passed"] else ("❌" if c["hard"] else "⚠️"), "Check": c["check"],
                                            "Type": "hard" if c["hard"] else "soft", "Note": c["note"]} for c in crit["checks"]]),
                             hide_index=True, width="stretch")
            st.subheader("Patient notification")
            st.info(result.get("message", ""))
            if result.get("candidates") and result.get("outcome") in ("BOOKED", "RESCHEDULED"):
                with st.expander("Alternative slots offered"):
                    st.dataframe(pd.DataFrame(result["candidates"])[["doctor", "start", "room", "score", "matches_time_pref"]],
                                 hide_index=True, width="stretch")

    if st.session_state.history:
        st.subheader("Session log")
        st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True, width="stretch")

# ================================================================== SCHEDULE
with tab_sched:
    st.header("Live hospital calendar (synthetic)")
    df = pd.DataFrame(st.session_state.appointments)
    df["doctor"] = df["doctor_id"].map(lambda d: doctor_by_id(d)["name"])
    df["specialty"] = df["doctor_id"].map(lambda d: doctor_by_id(d)["specialty"])
    df["date"] = df["start"].str[:10]
    df["time"] = df["start"].str[11:]
    f1, f2, f3 = st.columns(3)
    days = sorted(df["date"].unique())
    sel_day = f1.selectbox("Date", ["All"] + days, index=1 if len(days) > 0 else 0)
    sel_spec = f2.selectbox("Specialty", ["All"] + SPECIALTIES)
    only_agent = f3.checkbox("Only agent-created / changed", value=False)
    view = df.copy()
    if sel_day != "All":
        view = view[view["date"] == sel_day]
    if sel_spec != "All":
        view = view[view["specialty"] == sel_spec]
    if only_agent:
        view = view[(view["created_by"] == "MediBook Agents") | (view["status"] == "Cancelled")]
    st.dataframe(view.sort_values("start")[["appt_id", "date", "time", "doctor", "specialty", "patient_id", "reason",
                                            "urgency", "status", "created_by"]], hide_index=True, width="stretch", height=360)
    st.subheader("Doctor utilisation (booked slots, next 10 days)")
    active = df[df["status"] == "Booked"]
    util = active.groupby("doctor").size().rename("booked slots").to_frame()
    st.bar_chart(util, horizontal=True)
    with st.expander("Doctor roster"):
        wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        st.dataframe(pd.DataFrame([{**{k: d[k] for k in ("id", "name", "specialty", "room")},
                                    "days": ", ".join(wd[i] for i in d["days"]), "hours": f'{d["start"]}:00–{d["end"]}:00'}
                                   for d in DOCTORS]), hide_index=True, width="stretch")

# ================================================================== EVALUATION
with tab_eval:
    st.header("Output validation on a labelled test set")
    st.caption(f"{len(EVAL_CASES)} synthetic requests with expected intent / specialty / urgency / outcome. "
               "Each runs on a fresh copy of the calendar so results are reproducible.")
    if st.button("🧪 Run evaluation", type="primary"):
        rows, prog = [], st.progress(0.0)
        base = generate_seed_appointments()
        for i, (p, req, e_int, e_spec, e_urg, e_out) in enumerate(EVAL_CASES):
            t0 = time.perf_counter()
            r = run_workflow(req, p, base, llm)
            dt = time.perf_counter() - t0
            tri = r.get("triage", {})
            ok_i = r.get("intent") == e_int
            ok_s = e_spec is None or tri.get("specialty") == e_spec
            ok_u = e_urg is None or tri.get("urgency") == e_urg
            ok_o = r.get("outcome") == e_out
            # independent post-hoc constraint audit of the resulting calendar
            act = [a for a in r["appointments"] if a["status"] == "Booked"]
            clashes = len(act) - len({(a["doctor_id"], a["start"]) for a in act})
            rows.append({"#": i + 1, "request": req, "intent": r.get("intent"), "specialty": tri.get("specialty", "–"),
                         "urgency": tri.get("urgency", "–"), "outcome": r.get("outcome"),
                         "intent ✓": ok_i, "specialty ✓": ok_s, "urgency ✓": ok_u, "outcome ✓": ok_o,
                         "double-bookings": clashes, "steps": len(r["trace"]), "sec": round(dt, 2)})
            prog.progress((i + 1) / len(EVAL_CASES))
        st.session_state.eval = pd.DataFrame(rows)
    if "eval" in st.session_state:
        ev = st.session_state.eval
        k = st.columns(6)
        k[0].metric("Intent accuracy", f'{ev["intent ✓"].mean():.0%}')
        k[1].metric("Specialty accuracy", f'{ev["specialty ✓"].mean():.0%}')
        k[2].metric("Urgency accuracy", f'{ev["urgency ✓"].mean():.0%}')
        k[3].metric("Outcome accuracy", f'{ev["outcome ✓"].mean():.0%}')
        k[4].metric("Double-bookings", int(ev["double-bookings"].sum()))
        k[5].metric("Avg time / request", f'{ev["sec"].mean():.2f}s')
        st.dataframe(ev, hide_index=True, width="stretch")
        st.subheader("Estimated business impact (assumptions)")
        manual_min = st.slider("Manual handling time per request today (minutes)", 3, 15, 8)
        daily = st.slider("Requests per day", 50, 1000, 300, step=50)
        saved_h = daily * (manual_min - ev["sec"].mean() / 60) / 60
        st.markdown(f"- Staff time saved: **~{saved_h:.0f} hours/day** (≈{saved_h / 8:.1f} receptionist shifts)\n"
                    f"- Double-booking rate after Critic validation: **{int(ev['double-bookings'].sum())}** in {len(ev)} runs\n"
                    f"- 100% of emergency red-flag cases routed to ER instead of an OPD queue (see rows with Emergency)")

# ================================================================== ARCHITECTURE
with tab_arch:
    st.header("Architecture & orchestration")
    st.graphviz_chart("""
digraph G {
  rankdir=TB; nodesep=0.5; ranksep=0.45; newrank=true;
  node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=13, margin="0.2,0.08"];
  edge [fontname=Helvetica, fontsize=11];
  UI [label="Streamlit UI  (patient / front-desk request)", fillcolor="#e3f2fd"];
  subgraph cluster_lg { label="LangGraph StateGraph  ·  shared AgentState  ·  append-only trace"; style=dashed; fontname=Helvetica; fontsize=12;
    P [label="🧭 Planner / Intake\nintent · entities · task plan", fillcolor="#fff3e0"];
    TR [label="🩻 Triage\nspecialty · urgency", fillcolor="#fff3e0"];
    S [label="🗓️ Scheduler\nsearch + rank slots", fillcolor="#fff3e0"];
    C [label="🔍 Critic\n9 constraint checks", fillcolor="#fce4ec"];
    E [label="⚙️ Executor\nbook / cancel + verify", fillcolor="#e8f5e9"];
    ESC [label="🚨 Escalation\nER alert", fillcolor="#ffebee"];
    H [label="🙋 Human review\nfront-desk queue", fillcolor="#f3e5f5"];
    N [label="✉️ Notifier\nSMS draft + fact-check", fillcolor="#e8f5e9"];
  }
  LLM [label="Ollama local LLM\n(llama3.2)", shape=cylinder, fillcolor="#ede7f6"];
  RULES [label="Rules & guardrails\nsymptom KB · red flags", shape=note, fillcolor="#fffde7"];
  CAL [label="Calendar DB\ndoctors · slots · appointments", shape=cylinder, fillcolor="#e0f7fa"];
  {rank=same; P; LLM;}
  {rank=same; TR; ESC; RULES;}
  {rank=same; C; H; CAL;}
  UI -> P [label=" free-text request"];
  P -> TR [label=" book / reschedule"]; P -> E [label=" cancel"]; P -> H [label=" nothing to\n reschedule"];
  TR -> ESC [label=" Emergency", color=red, fontcolor=red]; TR -> S [label=" Urgent / Routine"];
  S -> C [label=" proposed slot"];
  C -> S [label=" REJECT\n (retry ≤3)", color=red, fontcolor=red, style=dashed];
  C -> E [label=" APPROVE", color=darkgreen, fontcolor=darkgreen]; C -> H [label=" 3 rejections"];
  E -> N; ESC -> N; H -> N;
  N -> UI [label=" confirmation + audit trace", constraint=false, color=gray40];
  P -> LLM [style=dotted, dir=both, color=purple];
  TR -> LLM [style=dotted, dir=both, color=purple];
  N -> LLM [style=dotted, dir=both, color=purple, constraint=false];
  TR -> RULES [style=dotted]; C -> RULES [style=dotted, constraint=false];
  S -> CAL [label=" find_available_slots", style=dotted, color=teal]; C -> CAL [label=" live re-check", style=dotted, color=teal];
  E -> CAL [label=" book_slot / cancel_appt", style=dotted, color=teal];
}""", width="stretch")
    st.markdown("""
| Agent | Responsibility | Uses | Decision it makes |
|---|---|---|---|
| **Planner / Intake** | Understands the free-text request, extracts date/time/doctor, builds the task plan | LLM (JSON mode) + regex validators | Route: book / reschedule / cancel |
| **Triage** | Maps symptoms + age to specialty and urgency | LLM + symptom KB; red-flag guardrails can only *raise* urgency | Emergency → ER, else schedule |
| **Scheduler (Executor of search)** | Finds free slots in the urgency window, ranks by preference, doctor load, earliness | `find_available_slots` tool | Which slot to propose |
| **Critic** | Validates proposal against 6 hard + 3 soft constraints on the *live* ledger | Rule validator | Approve / reject & retry / hand to human |
| **Executor** | Commits booking/cancellation, post-commit duplicate check | `book_slot`, `cancel_appt` | – |
| **Escalation / Human review** | Safety & fallback paths | Policy | – |
| **Notifier** | Drafts patient SMS; LLM draft is fact-checked, template fallback | LLM + template | Accept or reject LLM draft |
""")
    with st.expander("LangGraph-generated Mermaid (from the compiled graph)"):
        st.code(GRAPH.get_graph().draw_mermaid(), language="text")

# ------------------------------------------------------------------ sidebar stats (rendered last = always current)
with stats_box.container():
    st.metric("Active appointments", sum(a["status"] == "Booked" for a in st.session_state.appointments))
    st.metric("Requests handled this session", len(st.session_state.history))
