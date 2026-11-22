
import os
import sys
from dotenv import load_dotenv
from time import perf_counter

from langchain.messages import ToolMessage

# Add the parent directory (project root) to Python's path so it knows what "AI" is
# when you click the "Run" button in VS Code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.llm import Groq_llm
from AI.config import settings
llm=Groq_llm(settings.llm_model_name,settings.temperature,settings.groq_llm_api_key)

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



query = f"""what is the weather in trichy?"""

from langchain_core.utils.uuid import uuid7
agent_id=str(uuid7())


#--------------------langsmith tracing--------------------
print("Running create_agent...", flush=True)
load_dotenv()

if not os.getenv("LANGSMITH_API_KEY"):
    raise RuntimeError("LANGSMITH_API_KEY is missing")

import langsmith as ls
project_name=settings.project_name
environment=settings.APP_ENV
agent_version=settings.agent_version
started_at = perf_counter()
with ls.tracing_context(
    project_name=project_name,
    enabled=True,
    tags=[environment, project_name, agent_version],
    metadata={
        "session_id": agent_id,
        "environment": environment,
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
print(agent_result["messages"][-1].content_blocks[-1].get("text", "Something went wrong"))

#--------------------observability--------------------
print("Observability metrics...", flush=True)
from langchain_core.messages import AIMessage, ToolMessage
latency_seconds = perf_counter() - started_at
messages = agent_result["messages"]
ai_messages = [message for message in messages if isinstance(message, AIMessage)]

llm_call_count = len(ai_messages)
tool_call_count = sum(len(message.tool_calls or []) for message in ai_messages)
completed_tool_count = sum(isinstance(message, ToolMessage) for message in messages)

input_tokens = 0
output_tokens = 0

for message in ai_messages:
    usage = message.usage_metadata or message.response_metadata.get("token_usage", {})
    input_tokens += usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens += usage.get("output_tokens", usage.get("completion_tokens", 0))

print(f"LLM calls: {llm_call_count}")
print(f"Tool calls requested: {tool_call_count}")
print(f"Tool results returned: {completed_tool_count}")
print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Agent latency: {latency_seconds:.3f} seconds")

#--------------------agent memory for tracing--------------------
print("Tracing agent memory...", flush=True)
for message in agent_result["messages"]:
    if message.type == "ai":
        for block in message.content_blocks:
            if block["type"] == "reasoning":
                print("Reasoning:", block["reasoning"])
            elif block["type"] == "tool_call":
                print(f"Tool call: {block['name']}({block['args']})")
            elif block["type"] == "text":
                print("Assistant:", block["text"])

    elif message.type == "tool":
        print(f"Tool result [{message.name}]: {message.content}")

#--------------------deep agent--------------------
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


#--------------------streaming agent--------------------
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

