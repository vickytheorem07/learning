
import asyncio
import os
import sys
import selectors
from time import perf_counter
from urllib import request
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import langsmith as ls

# Load environment variables
load_dotenv()


# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AI.llm import Groq_llm
from AI.config import settings

# PostgreSQL connection URL
DB_URI = settings.memory_database_url

# Initialize LLM
llm = Groq_llm(
    settings.llm_model_name,
    settings.temperature,
    settings.groq_llm_api_key
)

# System prompt
system_prompt = """
You are a helpful assistant.
Use the get_weather tool whenever the user asks about weather.
"""

# Weather tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Retrieve conversation history
async def get_conversation_history(checkpointer, conversation_id):

    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }

    checkpoint = await checkpointer.aget(config)
    
    if not checkpoint:
        return []

    messages = checkpoint["channel_values"].get("messages", [])

    history = []

    for msg in messages:

        # User messages
        if msg.type == "human":
            history.append({
                "role": "user",
                "message": msg.content
            })

        # AI messages without tool calls
        elif msg.type == "ai" and not msg.tool_calls:

            if msg.content:
                history.append({
                    "role": "AI",
                    "message": msg.content
                })

    return history


# Main execution
async def main():

    async with AsyncPostgresSaver.from_conn_string(
        DB_URI
    ) as checkpointer:

        # Initialize checkpoint tables
        await checkpointer.setup()

        # Create LangGraph agent
        agent = create_agent(
            model=llm.llm,
            tools=[get_weather],
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )

        # Conversation ID
        conversation_id = "0001"

        config = {
            "run_name": "weather-agent",
            "configurable": {
                "thread_id": conversation_id
            }
        }

        #--------------------LangSmith tracing--------------------
        if not os.getenv("LANGSMITH_API_KEY"):
            raise RuntimeError("LANGSMITH_API_KEY is missing")
        project_name=settings.project_name
        environment=settings.APP_ENV
        agent_version=settings.agent_version
        started_at = perf_counter()
        with ls.tracing_context(
            project_name=project_name,
            enabled=True,
            tags=[environment, project_name, agent_version],
            metadata={
                "session_id": conversation_id,
                "environment": environment,
            }
        ):


            # First message
            agent_result = await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="what is the weather in trichy?"
                        )
                    ]
                },
                config=config
            )

            print("\nFirst response:")
            print(agent_result["messages"][-1].content)

            # Second message - same conversation
            # agent_result = await agent.ainvoke(
            #     {
            #         "messages": [
            #             HumanMessage(
            #                 content="What city did I ask about?"
            #             )
            #         ]
            #     },
            #     config=config
            # )

            # print("\nSecond response:")
            # print(agent_result["messages"][-1].content)
        
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
        # Retrieve complete conversation history
        history = await get_conversation_history(
            checkpointer,
            conversation_id
        )

        print("\nConversation History:")
        print(history)


 
if __name__ == "__main__":

    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        )
    )
