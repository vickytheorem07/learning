from langchain_groq import ChatGroq
from abc import ABC, abstractmethod


class llm(ABC):
    @abstractmethod
    def generate_response(self):
        pass


class Groq_llm(llm):
    def __init__(self,model_name,temperature,api_key):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def generate_response(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content


