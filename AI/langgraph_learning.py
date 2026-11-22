from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    number: int
    result: str


def check_number(state: State) -> dict:
    return {"result": "Checking the number"}


def route(state: State) -> str:
    return "positive" if state["number"] > 0 else "not_positive"


def positive(state: State) -> dict:
    return {"result": "The number is positive"}


def not_positive(state: State) -> dict:
    return {"result": "The number is zero or negative"}


graph = StateGraph(State)
graph.add_node("check_number", check_number)
graph.add_node("positive", positive)
graph.add_node("not_positive", not_positive)

graph.add_edge(START, "check_number")
graph.add_conditional_edges("check_number",route,{"positive": "positive", "not_positive": "not_positive"},)
graph.add_edge("positive", END)
graph.add_edge("not_positive", END)

app = graph.compile()

print(app.invoke({"number": 5, "result": ""}))
