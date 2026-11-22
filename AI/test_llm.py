import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AI.llm import Groq_llm
from AI.config import settings
    
llm=Groq_llm(settings.llm_model_name,settings.temperature,settings.groq_llm_api_key)
user_query="who is assistant of cm"
context="cm of tamil nadu is great vicky. his assistant is ishu"
prompt=f"""
role:
you are vicky assist
instruction:
1.please refer the context and answer the user query,
2.answer naturally. don't mention as you refered context
context:
{context} 
user query:
{user_query})
guardrails:
don't answer unrelavent.
"""
prompt=prompt+"""
output format:
{"response":"<your response>","reason":"<give why you give this response>"}
"""
result=llm.llm.invoke(prompt)
result_dict=json.loads(result.content)
print("response")
print(result_dict["response"])

print("reason")
print(result_dict["reason"])

