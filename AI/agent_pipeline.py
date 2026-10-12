
import os
import sys
from dotenv import load_dotenv

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


agent = create_agent(
    model=llm.llm,
    tools=[get_weather],
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)



query = f"""what is the weather like in trichy?"""

from langchain_core.utils.uuid import uuid7
agent_id=str(uuid7())

print("Running create_agent...", flush=True)
load_dotenv()

if not os.getenv("LANGSMITH_API_KEY"):
    raise RuntimeError("LANGSMITH_API_KEY is missing")

import langsmith as ls

with ls.tracing_context(
    project_name="email-agent-local",
    enabled=True,
    tags=["local", "email-agent", "v1.0"],
    metadata={
        "user_id": "user_id_001",
        "session_id": agent_id,
        "environment": "local",
    },
):
    agent_result = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config={
            "run_name": "email-agent",
            "configurable": {
                "thread_id": agent_id
            }
        },
    )
print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)
# print("Running create_deep_agent...", flush=True)
# from deepagents import create_deep_agent
# deep_agent = create_deep_agent(
#     model=llm.llm,
#     tools=[get_weather],
#     system_prompt=system_prompt,
#     checkpointer=checkpointer,
# )
# deep_agent_result = deep_agent.invoke(
#     {"messages": [{"role": "user", "content": query}]},
#     config={"configurable": {"thread_id": "weather-da"}},
# )
# print("\ncreate_deep_agent:")
# print(deep_agent_result["messages"][-1].content_blocks)


stream_agent_id=str(uuid7())
stream = agent.stream_events(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": stream_agent_id}},
    version="v3",
)

print("streaming...")
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
        "thread_id": stream_agent_id
    }
}
state = agent.get_state(config)

print("Agent memory:")
for message in state.values["messages"]:
    print(f"{message.type}: {message.content}")