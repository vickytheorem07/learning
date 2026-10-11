
import os
import sys

# Add the parent directory (project root) to Python's path so it knows what "AI" is
# when you click the "Run" button in VS Code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.llm import Groq_llm
from AI.config import Config
llm=Groq_llm(Config.llm_model_name,Config.temperature,Config.llm_api_key)

system_prompt="You are a helpful assistant. USE get_weather tool for getting weather information."

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

from langchain.tools import tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

from langchain.agents import create_agent
from deepagents import create_deep_agent

agent = create_agent(
    model=llm.llm,
    tools=[get_weather],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)

deep_agent = create_deep_agent(
    model=llm.llm,
    tools=[get_weather],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)

query = f"""what is the weather like in san francisco?"""

print("Running create_agent...", flush=True)
agent_result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": "weather-lc"}},
)
print("Running create_deep_agent...", flush=True)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": "weather-da"}},
)
print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)
print("\ncreate_deep_agent:")
print(deep_agent_result["messages"][-1].content_blocks)

stream = agent.stream_events(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": "weather-stream-2"}},
    version="v3",
)
from langchain.messages import AIMessage, HumanMessage
for snapshot in stream.values:
    # Each snapshot contains the full state at that point
    latest_message = snapshot["messages"][-1]
    if latest_message.content:
        if isinstance(latest_message, HumanMessage):
            print(f"User: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

# Read the final state once
print("\n\nFinal results:")
final_state = stream.output
for message in final_state["messages"]:
    if message.type == "tool":
        print(f"Tool result [{message.name}]: {message.content}")
        if getattr(message, "status", None) == "error":
            print("Tool execution failed.")
    elif message.type == "ai" and message.content:
        print(f"Assistant: {message.content}")

#agent memory
config = {
    "configurable": {
        "thread_id": "weather-stream-2"
    }
}
state = agent.get_state(config)

print("Agent memory:")
for message in state.values["messages"]:
    print(f"{message.type}: {message.content}")