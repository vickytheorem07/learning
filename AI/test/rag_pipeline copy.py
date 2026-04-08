from config import Config

class rag:
    def __init__(self):
        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.9,
            max_tokens=None,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectordb = Chroma(
            persist_directory=os.path.join(base_path, "chroma_db"),
            embedding_function=self.embedding_model,
        )

        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})

    def retrieve_docs(self, query):
        return self.retriever.invoke(query)

    def generate_answer(self, query):
        docs = self.retrieve_docs(query)

        context = "\n\n".join([
            f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
            for doc in docs
        ])

        prompt = rag_prompt.format(
            context=context,
            question=query
        )

        response = self.llm.invoke(prompt)
        return response.content, docs

rag=rag()
llm=