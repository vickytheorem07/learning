
from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import uuid4
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_chroma import Chroma


class VectorDB(ABC):

    @abstractmethod
    def add_documents(self, docs: List[Document]) -> List[str]:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        pass

    @abstractmethod
    def update_document(self, doc_id: str, doc: Document) -> None:
        pass

    @abstractmethod
    def delete_document(self, doc_id: str):
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        pass


class ChromaVectorDB(VectorDB):

    def __init__(self, embedding_model, collection_name:str, persist_dir: str):
        self.db = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

    # CREATE
    def add_documents(self, docs: List[Document]) -> List[str]:
        ids = [str(uuid4()) for _ in docs]

        return self.db.add_documents(
            documents=docs,
            ids=ids
        )

    # RETRIEVE
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        return self.db.similarity_search(
            query,
            k=k
        )

    # UPDATE
    def update_document(self, doc_id: str, doc: Document) -> None:
        self.db.update_documents(
            ids=[doc_id],
            documents=[doc]
        )

    # DELETE
    def delete_document(self, doc_id: str):
        self.db.delete(ids=[doc_id])

    # GET ALL
    def get_all(self) -> Dict[str, Any]:
        return self.db.get()
