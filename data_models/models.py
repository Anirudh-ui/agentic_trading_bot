from pydantic import BaseModel
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

# Renamed QuestionRequest to QueryModel for clarity in main.py's /query endpoint
class QueryModel(BaseModel): 
    question: str
    session_id: str
    
class SessionIDModel(BaseModel):
    session_id: str
    
# Other models you may have
class RagToolSchema(BaseModel):
    question:str