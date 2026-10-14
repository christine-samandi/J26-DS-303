"""The C1 pipeline as a LangGraph state machine. The only module that imports langgraph.

Every node is a plain function from `orchestrator.stages`; this file only wires them and
adds two things to each node: it skips the node once an earlier stage has failed, and it
records the node in `stages_run`. Replacing LangGraph means rewriting this file only.
"""

from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from orchestrator import stages
from orchestrator.models import OrchestratorResult, PipelineState, TaxRequest
from orchestrator.stages import Deps, StageError

LINEAR = (
    ("extract", stages.extract),
    ("normalise", stages.normalise),
    ("retrieve", stages.retrieve),
    ("calculate", stages.calculate_tax),
    ("explain", stages.explain),
    ("score", stages.score),
    ("assure", stages.assure),
)
RERUN_TARGETS = ("retrieve", "calculate", "explain")


def build_graph(deps: Deps):
    graph = StateGraph(PipelineState)
    nodes = (*LINEAR, ("regenerate", stages.regenerate), ("deliver", stages.deliver))
    for name, fn in nodes:
        graph.add_node(name, _node(name, fn, deps))

    graph.add_edge(START, LINEAR[0][0])
    for (a, _), (b, _) in zip(LINEAR, LINEAR[1:], strict=False):
        graph.add_edge(a, b)
    graph.add_conditional_edges(
        "assure", stages.after_assure, {"regenerate": "regenerate", "deliver": "deliver"}
    )
    graph.add_conditional_edges("regenerate", stages.rerun_from, {t: t for t in RERUN_TARGETS})
    graph.add_edge("deliver", END)
    return graph.compile()


def run(request: TaxRequest, deps: Deps) -> OrchestratorResult:
    final = build_graph(deps).invoke({"request": request, "attempt": 1, "stages_run": ()})
    return final["result"]


def _node(name: str, fn: Callable[[PipelineState, Deps], dict], deps: Deps):
    def node(state: PipelineState) -> dict:
        if state.get("error") and name != "deliver":
            return {}
        try:
            update = fn(state, deps)
        except StageError as e:
            update = {"error": f"{name}: {e}"}
        return {**update, "stages_run": (*state.get("stages_run", ()), name)}

    node.__name__ = name
    return node
