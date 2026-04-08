from langchain_community.embeddings import HuggingFaceEmbeddings

from abc import ABC, abstractmethod
class Embedding(ABC):
    @abstractmethod
    def embed_documents(self,docs):
        pass
    @abstractmethod
    def embed_query(self,query):
        pass
class Hugging_face_embedding(Embedding):
    def __init__(self,model_name,api_key=None):
        self.embedding = HuggingFaceEmbeddings(
            model_name= model_name)
    def embed_documents(self,docs):
        return self.embedding.embed_documents(docs)
    def embed_query(self,query):
        return self.embedding.embed_query(query)
           
    