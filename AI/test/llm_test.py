from dotenv import load_dotenv
import os

load_dotenv() 

print(os.getenv("groq_api"))
os.environ["GROQ_API_KEY"] = os.getenv("groq_api")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.9,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant",
    ),
    ("human", "who made largest fibonacci spiral as guinness world record?"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)