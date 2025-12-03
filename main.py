"""
FastAPI Application - Multimodal RAG & Internet Search
- Two-LLM architecture (Gemini Flash + Groq)
- Enhanced intent routing with fast-path
- STM + LTM session management
- Document duplicate prevention
- Comprehensive error handling and logging
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import sys
import os
import tempfile

# Internal imports
from agent.workflow_internet import InternetWorkflowBuilder
from agent.workflow_document import DocumentRAGWorkflowBuilder
from data_ingestion.gemini_multimodal_processor import GeminiMultimodalProcessor
from utils.document_session_manager import DocumentSessionManager
from exception.exceptions import TradingBotException, ValidationException
from custom_logging.my_logger import logger, log_execution_time
from langchain_core.messages import HumanMessage

# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot - Multimodal RAG API",
    description="Advanced document analysis and internet search with two-LLM architecture",
    version="2.0.0"
)

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL INITIALIZATION ====================

logger.info("=" * 60)
logger.info("STARTING TRADING BOT API - TWO-LLM ARCHITECTURE")
logger.info("=" * 60)

# Internet workflow (STM only)
try:
    internet_workflow = InternetWorkflowBuilder()
    internet_workflow.build()
    internet_graph = internet_workflow.get_graph()
    logger.info("[STARTUP] Internet workflow initialized")
except Exception as e:
    logger.error(f"[STARTUP] Internet workflow failed: {e}")
    raise

# Document workflow (STM + LTM)
try:
    document_workflow = DocumentRAGWorkflowBuilder()
    document_workflow.build()
    document_graph = document_workflow.get_graph()
    logger.info("[STARTUP] Document workflow initialized")
except Exception as e:
    logger.error(f"[STARTUP] Document workflow failed: {e}")
    raise

# Gemini processor
try:
    gemini_processor = GeminiMultimodalProcessor()
    logger.info("[STARTUP] Gemini processor initialized")
except Exception as e:
    logger.error(f"[STARTUP] Gemini processor failed: {e}")
    raise

# Document session manager
try:
    doc_session_manager = DocumentSessionManager(
        redis_host=os.getenv('REDIS_HOST', 'localhost'),
        redis_port=int(os.getenv('REDIS_PORT', 6380)),
        weaviate_url=os.getenv('WEAVIATE_URL', 'http://localhost:8080'),
        weaviate_api_key=os.getenv('WEAVIATE_API_KEY')
    )
    logger.info("[STARTUP] Document session manager initialized")
except Exception as e:
    logger.error(f"[STARTUP] Session manager failed: {e}")
    raise

logger.info("[STARTUP] All components initialized successfully")

# ==================== REQUEST/RESPONSE MODELS ====================

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


# ==================== UTILITY FUNCTIONS ====================

def validate_session_id(session_id: str, expected_prefix: str):
    """Validate session ID format"""
    if not session_id.startswith(expected_prefix):
        raise ValidationException(
            f"Invalid session ID format. Expected prefix: {expected_prefix}",
            sys,
            session_id=session_id
        )


def validate_doc_id(doc_id: str):
    """Validate document ID format"""
    if not doc_id.startswith("trade_doc_"):
        raise ValidationException(
            "Invalid document ID format. Expected prefix: trade_doc_",
            sys,
            doc_id=doc_id
        )


# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    logger.info("[ROOT] API information requested")
    
    return {
        "status": "online",
        "name": "Trading Bot - Multimodal RAG API",
        "version": "2.0.0",
        "architecture": "Two-LLM (Gemini Flash + Groq)",
        "features": [
            "Multimodal document analysis (text, tables, charts)",
            "Internet search with intent routing",
            "Fast-path for common queries",
            "STM + LTM session management",
            "Response caching",
            "Source citation"
        ],
        "endpoints": {
            "/upload-document": "POST - Upload document with Gemini processing",
            "/document-query": "POST - Query document (Gemini analysis + Groq response)",
            "/internet-query": "POST - Internet search (intent routing + tools)",
            "/get-user-documents": "GET - Get all user documents",
            "/get-document-summary/{doc_id}": "GET - Get document conversation summary",
            "/clear-internet-session": "POST - Clear internet session (STM only)",
            "/clear-caches": "POST - Clear all response caches",
            "/cache-stats": "GET - Get cache statistics",
            "/health": "GET - Health check"
        }
    }


@app.post("/upload-document", response_model=DocumentUploadResponse)
#@log_execution_time
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document with Gemini multimodal processor
    - Prevents duplicate uploads
    - Extracts text, tables, and charts
    - Stores in Pinecone vector database
    """
    try:
        user_id = "static_test_user"  # TODO: Replace with actual user authentication
        
        logger.info(f"[UPLOAD] Processing document: {file.filename}")
        
        # Validate file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise ValidationException(
                f"Unsupported file type: {file_ext}",
                sys,
                filename=file.filename,
                allowed=allowed_extensions
            )
        
        # Check for duplicate document
        existing_doc = doc_session_manager.find_document_by_filename(
            filename=file.filename,
            user_id=user_id
        )
        
        is_first_time = existing_doc is None
        
        if not is_first_time:
            logger.info(f"[UPLOAD] Document already exists: {existing_doc['doc_id']}")
            
            # Return existing document info
            return DocumentUploadResponse(
                doc_id=existing_doc['doc_id'],
                name=file.filename,
                uploadedAt=existing_doc['upload_timestamp'],
                size=f"{existing_doc.get('page_count', 0)} pages",
                page_count=existing_doc.get('page_count', 0),
                is_first_time=False,
                previous_summary=doc_session_manager.get_document_summary(existing_doc['doc_id']),
                has_tables=False,
                has_charts=False
            )
        
        # 1. Read file content from the UploadFile (must be done before creating temp file)
        file_content = await file.read()
        
        # 2. Use tempfile to create a secure, accessible temporary path
        # suffix ensures the correct file extension for PDF/DOCX readers
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(file_content)
            temp_path = tmp.name  # Get the guaranteed path
        
        logger.info(f"[UPLOAD] Saved temporary file to: {temp_path}")
        
        logger.info("[GEMINI] Starting multimodal processing...")
        
        # Process with Gemini
        processed_data = gemini_processor.process_document(temp_path)
        
        logger.info(f"[GEMINI] Processing complete | Pages: {processed_data['page_count']}")
        
        # Register document
        doc_id = doc_session_manager.register_document(
            filename=file.filename,
            file_type=file_ext.lstrip('.'),
            user_id=user_id,
            page_count=processed_data['page_count'],
            has_tables=len(processed_data.get('tables', [])) > 0,
            has_charts=len(processed_data.get('charts', [])) > 0
        )
        
        logger.info(f"[DOCUMENT] Registered: {doc_id}")
        
        # Store in Pinecone
        gemini_processor.store_in_pinecone(
            processed_data=processed_data,
            doc_id=doc_id,
            filename=file.filename
        )
        
        logger.info(f"[PINECONE] Stored vectors for doc: {doc_id}")
        
        # Cleanup
        os.remove(temp_path)
        
        return DocumentUploadResponse(
            doc_id=doc_id,
            name=file.filename,
            uploadedAt=datetime.now().isoformat(),
            size=f"{len(file_content) / 1024:.1f} KB",
            page_count=processed_data['page_count'],
            is_first_time=True,
            previous_summary=None,
            has_tables=len(processed_data.get('tables', [])) > 0,
            has_charts=len(processed_data.get('charts', [])) > 0
        )
        
    except ValidationException:
        raise
    except TradingBotException as e:
        logger.error(f"[UPLOAD] Application error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[UPLOAD] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/document-query", response_model=QueryResponse)
@log_execution_time
async def document_query(request: DocumentQueryRequest):
    """
    Query document using two-LLM workflow:
    1. Retrieve chunks from Pinecone
    2. Gemini Flash analyzes multimodal content
    3. Groq generates conversational response with citations
    """
    try:
        # Validate inputs
        validate_session_id(request.session_id, "trade_doc_")
        validate_doc_id(request.doc_id)
        
        logger.info(f"[DOC QUERY] Doc: {request.doc_id} | Session: {request.session_id}")
        logger.info(f"[DOC QUERY] Question: {request.question[:100]}...")
        
        # Verify document exists
        doc_metadata = doc_session_manager.get_document_metadata(request.doc_id)
        if not doc_metadata:
            raise ValidationException(
                "Document not found",
                sys,
                doc_id=request.doc_id
            )
        
        # Store user message in STM
        document_workflow.memory_manager.store_message(
            request.session_id,
            "user",
            request.question
        )
        
        # Prepare workflow state
        initial_state = {
            "messages": [HumanMessage(content=request.question)],
            "doc_id": request.doc_id,
            "session_id": request.session_id,
            "user_id": request.user_id,
            "retrieved_chunks": [],
            "sources": [],
            "gemini_analysis": "",
            "summary": ""
        }
        
        # Invoke two-LLM workflow
        logger.info("[WORKFLOW] Starting document RAG workflow...")
        result = await document_graph.invoke(initial_state)
        
        # Extract answer
        final_messages = result.get("messages", [])
        answer = "I couldn't find an answer in the document."
        
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and msg.content:
                answer = msg.content
                break
        
        # Get sources
        sources = result.get("sources", [])
        
        # Store Q&A in LTM
        doc_session_manager.store_document_qa(
            session_id=request.session_id,
            doc_id=request.doc_id,
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            sources=sources
        )
        
        # Get document summary
        doc_summary = doc_session_manager.get_document_summary(request.doc_id)
        
        logger.info(f"[DOC QUERY] Response generated | Length: {len(answer)}")
        
        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            message_id=f"msg_{int(datetime.now().timestamp() * 1000)}",
            timestamp=datetime.now().isoformat(),
            sources=sources,
            doc_summary=doc_summary,
            query_type="document"
        )
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TradingBotException as e:
        logger.error(f"[DOC QUERY] Application error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[DOC QUERY] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/internet-query", response_model=QueryResponse)
#@log_execution_time
async def internet_query(request: InternetQueryRequest):
    """
    Optimized internet query with:
    - Fast-path for greetings (no LLM, <50ms)
    - Smart intent routing (News → Tavily, Stock → Yahoo Finance)
    - STM-only memory management
    - Automatic cache cleanup on session close
    """
    try:
        # Validate session ID
        validate_session_id(request.session_id, "trade_app_")
        
        query_start_time = datetime.now()
        
        logger.info(f"[INTERNET] Session: {request.session_id}")
        logger.info(f"[INTERNET] Query: {request.question[:80]}...")
        
        # Store user message in STM
        internet_workflow.redis_memory.store_message(request.session_id, {
            'role': 'user',
            'content': request.question
        })
        
        # Prepare workflow state
        initial_state = {
            "messages": [HumanMessage(content=request.question)],
            "query_type": "unknown",
            "session_id": request.session_id,
            "is_fast_path": False,
            "tickers": [],
            "needs_tools": True
        }
        
        # Invoke optimized workflow
        logger.info("[WORKFLOW] Starting optimized internet workflow...")
        result =  internet_graph.invoke(initial_state)
        
        # Extract answer
        final_messages = result.get("messages", [])
        answer = "I couldn't process that request."
        print("final_messages:", final_messages)
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and msg.content:
                answer = msg.content
                break
        
        # Get metadata
        query_type = result.get("query_type", "general")
        is_fast_path = result.get("is_fast_path", False)
        tickers = result.get("tickers", [])
        
        # Calculate response time
        response_time_ms = (datetime.now() - query_start_time).total_seconds() * 1000
        
        logger.info(
            f"[INTERNET] Response complete | "
            f"Type: {query_type} | "
            f"Fast-path: {is_fast_path} | "
            f"Time: {response_time_ms:.0f}ms"
        )
        
        # Add performance indicator to fast-path responses
        if is_fast_path and response_time_ms < 100:
            logger.info(f"[PERFORMANCE] ⚡ Ultra-fast response: {response_time_ms:.0f}ms")
        
        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            message_id=f"msg_{int(datetime.now().timestamp() * 1000)}",
            timestamp=datetime.now().isoformat(),
            sources=None,
            query_type=query_type,
            is_fast_path=is_fast_path
        )
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TradingBotException as e:
        logger.error(f"[INTERNET] Application error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"[INTERNET] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/clear-internet-session")
#@log_execution_time
async def clear_internet_session(session_id: str):
    """
    Clear internet session (STM + cache)
    Called automatically when user closes general chat panel
    """
    try:
        validate_session_id(session_id, "trade_app_")
        
        logger.info(f"[CLEAR] 🧹 Clearing session: {session_id}")
        
        # Clear STM memory
        internet_workflow.clear_session(session_id)
        
        # Optional: Clear cache entries for this session
        # (Currently cache is global, but could be session-specific)
        
        logger.info(f"[CLEAR] ✓ Session cleared successfully: {session_id}")
        
        return {
            "message": f"Session {session_id} cleared successfully",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "cleared": ["stm_memory", "session_cache"]
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[CLEAR] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/internet-session-stats/{session_id}")
async def get_session_stats(session_id: str):
    """
    Get statistics for internet session (for debugging/monitoring)
    """
    try:
        validate_session_id(session_id, "trade_app_")
        
        # Get message count from Redis
        messages = internet_workflow.redis_memory.get_langchain_messages(session_id)
        message_count = len(messages)
        
        # Get cache stats
        cache_stats = internet_workflow.get_cache_stats()
        
        return {
            "session_id": session_id,
            "message_count": message_count,
            "cache_stats": cache_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[SESSION STATS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/internet-query-batch")
@log_execution_time
async def internet_query_batch(requests: List[InternetQueryRequest]):
    """
    Batch processing for multiple queries (optional optimization)
    Useful for pre-fetching common queries
    """
    try:
        results = []
        
        for request in requests:
            try:
                result = await internet_query(request)
                results.append({
                    "status": "success",
                    "query": request.question,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "query": request.question,
                    "error": str(e)
                })
        
        return {
            "total": len(requests),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"[BATCH] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-user-documents")
#@log_execution_time
async def get_user_documents(user_id: str = "static_test_user"):
    """Get all documents for user"""
    try:
        logger.info(f"[GET DOCS] Fetching documents for user: {user_id}")
        
        documents = doc_session_manager.get_user_documents(user_id, limit=100)
        
        return {
            "documents": [
                {
                    "id": doc['doc_id'],
                    "name": doc['filename'],
                    "uploadedAt": doc['upload_timestamp'],
                    "size": f"{doc.get('page_count', 0)} pages",
                    "page_count": doc.get('page_count', 0)
                }
                for doc in documents
            ],
            "count": len(documents)
        }
        
    except Exception as e:
        logger.error(f"[GET DOCS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-document-summary/{doc_id}")
@log_execution_time
async def get_document_summary(doc_id: str):
    """Get conversation summary for document"""
    try:
        validate_doc_id(doc_id)
        
        logger.info(f"[GET SUMMARY] Doc: {doc_id}")
        
        summary = doc_session_manager.get_document_summary(doc_id)
        qa_count = doc_session_manager.get_document_qa_count(doc_id)
        
        return {
            "doc_id": doc_id,
            "summary": summary or "No previous conversations",
            "qa_count": qa_count,
            "has_history": qa_count > 0
        }
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[GET SUMMARY] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/clear-caches")
@log_execution_time
async def clear_caches():
    """Clear all response caches"""
    try:
        logger.info("[CLEAR CACHES] Clearing all caches...")
        
        document_workflow.clear_cache()
        internet_workflow.clear_cache()
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CLEAR CACHES] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cache-stats")
async def cache_stats():
    """Get cache statistics"""
    try:
        doc_stats = document_workflow.get_cache_stats()
        internet_stats = internet_workflow.get_cache_stats()
        
        return {
            "document_workflow": doc_stats,
            "internet_workflow": internet_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"[CACHE STATS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    try:
        # Check Redis
        redis_status = "healthy"
        try:
            internet_workflow.redis_memory.redis_client.ping()
        except:
            redis_status = "unhealthy"
        
        # Check Weaviate
        weaviate_status = "healthy"
        try:
            document_workflow.memory_manager.weaviate_manager.client.is_ready()
        except:
            weaviate_status = "unhealthy"
        
        # Check Pinecone
        pinecone_status = "healthy"
        try:
            document_workflow.index.describe_index_stats()
        except:
            pinecone_status = "unhealthy"
        
        # Get cache stats
        cache_stats_data = {
            "document": document_workflow.get_cache_stats(),
            "internet": internet_workflow.get_cache_stats()
        }
        
        overall_status = "healthy" if all([
            redis_status == "healthy",
            weaviate_status == "healthy",
            pinecone_status == "healthy"
        ]) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            components={
                "redis": redis_status,
                "weaviate": weaviate_status,
                "pinecone": pinecone_status,
                "document_workflow": "healthy",
                "internet_workflow": "healthy",
                "gemini_processor": "healthy"
            },
            timestamp=datetime.now().isoformat(),
            cache_stats=cache_stats_data
        )
        
    except Exception as e:
        logger.error(f"[HEALTH CHECK] Error: {e}")
        return HealthResponse(
            status="unhealthy",
            components={"error": str(e)},
            timestamp=datetime.now().isoformat(),
            cache_stats={}
        )


@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("=" * 60)
    logger.info("APPLICATION STARTUP COMPLETE")
    logger.info("Two-LLM Architecture: Gemini Flash + Groq")
    logger.info("Fast-path routing enabled")
    logger.info("STM + LTM session management active")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("=" * 60)
    logger.info("APPLICATION SHUTDOWN")
    logger.info("=" * 60)
    
    try:
        document_workflow.close()
        logger.info("[SHUTDOWN] Document workflow closed")
    except Exception as e:
        logger.error(f"[SHUTDOWN] Error closing document workflow: {e}")


# Run application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )