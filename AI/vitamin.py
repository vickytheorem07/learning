import os
import sys

# Add the parent directory (project root) to Python's path so it knows what "AI" is
# when you click the "Run" button in VS Code.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AI.config import Config
from AI.llm import Groq_llm

llm=Groq_llm(Config.llm_model_name,Config.temperature,Config.llm_api_key)
print(llm.generate_response("What is the capital of France?"))