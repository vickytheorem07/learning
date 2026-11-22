from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFaceEmbeddings


class Embedding(ABC):
    @abstractmethod
    def embed_documents(self,docs) -> list[list[float]]:
        pass
    @abstractmethod
    def embed_query(self,query) -> list[float]:
        pass
class Hugging_face_embedding(Embedding):
    def __init__(self, model_name, api_key=None):
        model_kwargs = {"token": api_key} if api_key else {}
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
        )

    def embed_documents(self, docs):
        return self.embedding.embed_documents(docs)

    def embed_query(self, query):
        return self.embedding.embed_query(query)
