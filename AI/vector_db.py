from langchain_community.vectorstores import Chroma
from abc import ABC, abstractmethod


class Vector_db(ABC):
    @abstractmethod
    def add_documents(self,docs):
        pass
    @abstractmethod
    def similarity_search(self,query,k):
        pass
    @abstractmethod
    def update_document(self,doc):
        pass
    @abstractmethod
    def delete_document(self,doc_id):
        pass
    @abstractmethod
    def get_all(self):
        pass

class Chroma_vector_db(Vector_db):
    def __init__(self, embedding_model,PERSIST_DIR):
        self.db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model
        )

    # CREATE / INSERT
    def add_documents(self, docs):
        ids = [doc.metadata["source"] for doc in docs]
        self.db.add_documents(documents=docs, ids=ids)
        self.db.persist()

    # RETRIEVE
    def similarity_search(self, query, k):
        return self.db.similarity_search(query, k=k)

    # UPDATE
    def update_document(self, doc):
        doc_id = doc.metadata["source"]
        self.db.delete(ids=[doc_id])
        self.db.add_documents([doc], ids=[doc_id])
        self.db.persist()

    # DELETE
    def delete_document(self, doc_id):
        self.db.delete(ids=[doc_id])

    # GET
    def get_all(self):
        return self.db.get()
