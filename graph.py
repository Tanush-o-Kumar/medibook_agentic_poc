"""LangGraph orchestration: wires the agents into a stateful graph with
conditional routing and a Critic -> Scheduler feedback loop."""
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import StateGraph, START, END

import agents as A


class AgentState(TypedDict, total=False):
    request: str
    patient_id: str
    appointments: list
    intent: str
    extracted: dict
    plan: list
    existing: list
    triage: dict
    candidates: list
    proposed: Any
    rejected: list
    critic: dict
    attempts: int
    booking: Any
    cancelled: Any
    outcome: str
    message: str
    trace: Annotated[list, operator.add]   # every agent appends to the shared log


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", A.planner_agent)
    g.add_node("triage", A.triage_agent)
    g.add_node("scheduler", A.scheduler_agent)
    g.add_node("critic", A.critic_agent)
    g.add_node("executor", A.executor_agent)
    g.add_node("escalation", A.escalation_agent)
    g.add_node("human_review", A.human_review_agent)
    g.add_node("notifier", A.notifier_agent)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", A.route_after_planner,
                            {"triage": "triage", "executor": "executor", "human_review": "human_review"})
    g.add_conditional_edges("triage", A.route_after_triage, {"escalation": "escalation", "scheduler": "scheduler"})
    g.add_edge("scheduler", "critic")
    g.add_conditional_edges("critic", A.route_after_critic,
                            {"executor": "executor", "scheduler": "scheduler", "human_review": "human_review"})
    g.add_edge("executor", "notifier")
    g.add_edge("escalation", "notifier")
    g.add_edge("human_review", "notifier")
    g.add_edge("notifier", END)
    return g.compile()


GRAPH = build_graph()


def run_workflow(request, patient_id, appointments, llm, simulate_conflict=False, now=None):
    state = {"request": request, "patient_id": patient_id, "appointments": list(appointments),
             "rejected": [], "attempts": 0, "trace": []}
    cfg = {"configurable": {"llm": llm, "simulate_conflict": simulate_conflict, "now": now}, "recursion_limit": 40}
    return GRAPH.invoke(state, config=cfg)
