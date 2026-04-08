import os
import sys

# Add the parent directory (project root) to Python's path so it knows what "AI" is
# when you click the "Run" button in VS Code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.config import Config
from AI.llm import Groq_llm
from AI.embedding import Hugging_face_embedding
from AI.vector_db import Chroma_vector_db
from AI.loaders import DocumentLoader

class Rag:
    def __init__(self,llm,embedding,vector_db):
        self.llm = llm
        self.embedding=embedding
        self.vector_db=vector_db
    def adding_documents(self,docs):
        self.vector_db.add_documents(docs)
    def llm_response(self,query,top_k):
        relavant_docs=self.vector_db.similarity_search(query,k=top_k) 
        context = "\n\n".join([
            f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in relavant_docs
        ])
        prompt=f"""
        You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
        """
        return self.llm.invoke(prompt).content
        

llm=Groq_llm(Config.llm_model_name,Config.temperature,Config.llm_api_key)
embedding=Hugging_face_embedding(Config.embedding_model_name,Config.embedding_api_key)
vector_db=Chroma_vector_db(embedding.embedding,Config.persist_dir)
rag_instance=Rag(llm.llm,embedding.embedding,vector_db)
#documents=DocumentLoader().load_documents(Config.data_path)
#rag_instance.adding_documents(documents)
print(rag_instance.llm_response("What is the capital of France?",3))


