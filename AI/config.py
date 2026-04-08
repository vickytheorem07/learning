import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    llm_api_key = os.getenv("groq_api")
    llm_model_name = "qwen/qwen3-32b"
    temperature = 0.9
    
    embedding_api_key = os.getenv("huggingface_api")
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    data_path = r"\data"
    persist_dir = r"\chroma_db"
    top_k = 3