# from langchain.agents import create_agent

# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

# agent = create_agent(
#     model="openai:gpt-5.5",
#     tools=[get_weather],
#     system_prompt="You are a helpful assistant",
# )

# result = agent.invoke(
#     {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
# )
# print(result["messages"][-1].content_blocks)
import os
import sys

# Add the parent directory (project root) to Python's path so it knows what "AI" is
# when you click the "Run" button in VS Code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.llm import Groq_llm
from AI.config import Config
llm=Groq_llm(Config.llm_model_name,Config.temperature,Config.llm_api_key)

system_prompt="You are a helpful assistant"

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