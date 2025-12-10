from pydantic import BaseModel , Field
from typing import List
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict , Optional

# Renamed QuestionRequest to QueryModel for clarity in main.py's /query endpoint
class QueryModel(BaseModel): 
    question: str
    session_id: str
    
class SessionIDModel(BaseModel):
    session_id: str
    
# Other models you may have
class RagToolSchema(BaseModel):
    question:str


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    doc_id: str = Field(..., description="Unique document ID")
    name: str = Field(..., description="Document filename")
    uploadedAt: str = Field(..., description="Upload timestamp")
    size: str = Field(..., description="Document size")
    page_count: int = Field(..., description="Number of pages")
    is_first_time: bool = Field(..., description="First time uploading this document")
    previous_summary: Optional[str] = Field(None, description="Summary of previous conversations")
    has_tables: bool = Field(False, description="Document contains tables")
    has_charts: bool = Field(False, description="Document contains charts")

class DocumentInfo(BaseModel):
    """Model for listing user documents in the sidebar"""
    doc_id: str = Field(..., description="Unique document ID")
    name: str = Field(..., description="Document filename")
    uploadedAt: str = Field(..., description="Upload timestamp")
    page_count: int = Field(..., description="Number of pages")
    has_tables: bool = Field(..., description="Document contains tables")
    has_charts: bool = Field(..., description="Document contains charts")
    size: str = Field(..., description="Human-readable size (KB/MB)")
    summary: Optional[str] = Field(None, description="Summary snippet")
    
class DocumentQueryRequest(BaseModel):
    """Request model for document query"""
    question: str = Field(..., min_length=1, description="User question about document")
    session_id: str = Field(..., description="Session ID (format: trade_doc_{doc_id})")
    doc_id: str = Field(..., description="Document ID")
    user_id: Optional[str] = Field("static_test_user", description="User ID")


class InternetQueryRequest(BaseModel):
    """Request model for internet query"""
    question: str = Field(..., min_length=1, description="User query for internet search")
    session_id: str = Field(..., description="Session ID (format: trade_app_{uuid})")


class QueryResponse(BaseModel):
    """Response model for queries"""
    answer: str = Field(..., description="Generated answer")
    session_id: str = Field(..., description="Session ID")
    message_id: str = Field(..., description="Message ID")
    timestamp: str = Field(..., description="Response timestamp")
    sources: Optional[List[dict]] = Field(None, description="Source citations")
    doc_summary: Optional[str] = Field(None, description="Document conversation summary")
    query_type: Optional[str] = Field(None, description="Query type classification")
    is_fast_path: Optional[bool] = Field(False, description="Fast-path response flag")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    components: dict
    timestamp: str
    cache_stats: dict


class UserDocument(BaseModel):
    id: str = Field(..., description="Document ID")
    name: str = Field(..., description="File name")
    ploadedAt: str = Field(..., description="Upload timestamp")
    size: str = Field(..., description="Size string e.g. '120.5 KB'")
    page_count: int = Field(..., description="Page count")
    has_tables: bool = Field(..., description="Contains tables")
    has_charts: bool = Field(..., description="Contains charts")
    summary: Optional[str] = Field(None, description="Optional saved summary snippet")
        
class UserDocumentsResponse(BaseModel):
    documents: List[UserDocument]
    count: int

class DocumentSessionCreateRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID")
    user_id: Optional[str] = Field("static_test_user", description="User ID")

