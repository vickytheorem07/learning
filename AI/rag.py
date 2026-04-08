# =========================
# 1. ENV + LLM SETUP
# =========================
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("groq_api")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=None,
)

# =========================
# 2. LOAD TEXT FILES
# =========================
from langchain_community.document_loaders import TextLoader

base_path = os.path.dirname(__file__)

file_paths = [
    os.path.join(base_path, "file1.txt"),
    os.path.join(base_path, "file2.txt"),
    os.path.join(base_path, "file3.txt"),
    os.path.join(base_path, "file4.txt"),
    os.path.join(base_path, "file5.txt")
]

documents = []

for path in file_paths:
    loader = TextLoader(path)
    docs = loader.load()

    filename = os.path.basename(path)

    for d in docs:
        d.metadata = {
            "source": filename,
            "file_name": filename
        }

    documents.extend(docs)

# =========================
# 3. CHUNKING (1 file = 1 chunk)
# =========================
chunks = documents

# ---- OPTIONAL (COMMENTED): Overlap chunking ----
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
"""

print("Total chunks:", len(chunks))  # should be 5

# =========================
# 4. EMBEDDINGS + CHROMA DB
# =========================
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

persist_dir = os.path.join(base_path, "chroma_db")

# Explicit IDs = filenames
ids = [doc.metadata["source"] for doc in chunks]

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    ids=ids,
    persist_directory=persist_dir
)

vectordb.persist()

# =========================
# VERIFY STORAGE
# =========================
data = vectordb.get()
print("Stored IDs:", data["ids"])
print("Stored Metadata:", data["metadatas"])

# =========================
# ---- OPTIONAL (COMMENTED): CRUD OPERATIONS ----
# =========================
"""
# CREATE / INSERT
vectordb.add_documents(
    documents=new_docs,
    ids=["file6.txt"]
)

# RETRIEVE
results = vectordb.similarity_search("query", k=3)

# UPDATE (delete + insert)
vectordb.delete(ids=["file1.txt"])
vectordb.add_documents(updated_docs, ids=["file1.txt"])

# DELETE
vectordb.delete(ids=["file2.txt"])

# GET SPECIFIC
vectordb.get(ids=["file3.txt"])

# GET ALL
vectordb.get()
"""

# =========================
# 5. RETRIEVAL
# =========================
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

def retrieve_docs(query):
    return retriever.invoke(query)

# =========================
# 6. RAG PROMPT + LLM
# =========================
from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""
)

def generate_answer(query):
    docs = retrieve_docs(query)

    # Combine context
    context = "\n\n".join([
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc in docs
    ])

    prompt = rag_prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(prompt)
    return response.content, docs

# =========================
# 7. OUTPUT
# =========================
query = "hi"

answer, docs = generate_answer(query)

print("\nQ:", query)
print("\nA:", answer)

print("\nSources used:")
for d in docs:
    print("-", d.metadata["source"])