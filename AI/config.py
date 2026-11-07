# import os
# from dotenv import load_dotenv

# load_dotenv()

# class Config:
#     llm_api_key = os.getenv("groq_api")
#     llm_model_name = "openai/gpt-oss-20b"
#     #qwen/qwen3.8-27b
#     #qwen/qwen3-32b
#     #openai/gpt-oss-120b
#     #llm_model_name = "ometa-llama/llama-4-scout-17b-16e-instruct"
#     #openai/gpt-oss-20b
#     temperature = 0.9
    
#     embedding_api_key = os.getenv("huggingface_api")
#     embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

#     data_path = r"data"
#     persist_dir = r"chroma_db"
#     top_k = 3

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    groq_llm_api_key: str
    llm_model_name: str = "openai/gpt-oss-20b"
    temperature: float = 0.9
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    data_path: str = "AI/data"
    persist_dir: str = "chroma_db"
    top_k: int = 3
    LANGSMITH_API_KEY: str
    


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()