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
import json
# Internal imports
from agent.workflow_internet import InternetWorkflowBuilder
from agent.workflow_document import DocumentRAGWorkflowBuilder
from data_ingestion.gemini_multimodal_processor import GeminiMultimodalProcessor
from utils.document_session_manager import DocumentSessionManager
from exception.exceptions import TradingBotException, ValidationException
from custom_logging.my_logger import logger, log_execution_time
from langchain_core.messages import HumanMessage , AIMessage
from utils.postgres_manager import PostgresManager
from data_models.models import * 
from pinecone import Pinecone
from utils.session_registry import DocumentSessionRegistry
from utils.document_intent_classifier import DocumentIntentClassifier
from utils.memory_manager import HybridMemoryManager
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
classifier = DocumentIntentClassifier()
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
#postgres manager
try:
    postgres_manager = PostgresManager()
    logger.info("[STARTUP] postgres processor initialized")
except Exception as e:
    logger.error(f"[STARTUP] postgres processor failed: {e}")
    raise
# Document session manager
try:
    memory_manager = HybridMemoryManager(
         redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", 6380)),
            weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),    
            )
    logger.info("[MEMORY] HybridMemoryManager ready")               

except Exception as e:
        logger.error(f"[MEMORY INIT ERROR] {e}")
        raise
try:
    doc_session_manager = DocumentSessionManager(
        memory_manager=memory_manager
    )
    logger.info("[STARTUP] Document session manager initialized")
except Exception as e:
    logger.error(f"[STARTUP] Session manager failed: {e}")
    raise

logger.info("[STARTUP] All components initialized successfully")

try:
    doc_session_registry = DocumentSessionRegistry(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6380))
    )
    logger.info("[STARTUP] Document session registry initialized")

except Exception as e:
    logger.error(f"[STARTUP] Document session registry failed: {e}")
    raise

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
            "/delete-document/{doc_id}": "DELETE - Delete document and associated data",
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
async def upload_document(file: UploadFile = File(...)):
    """
    Handles document upload:
    - validates type
    - checks duplicates using file hash
    - processes document with Gemini (tables/charts/text)
    - stores chunks in Pinecone
    - writes metadata to Postgres
    """
    try:
        user_id = "static_test_user"
        logger.info(f"[UPLOAD] File received → {file.filename}")

        # -----------------------------------------------------------
        # 1. FILE VALIDATION
        # -----------------------------------------------------------
        allowed = [".pdf", ".docx", ".txt"]
        ext = os.path.splitext(file.filename)[1].lower()

        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )

        # -----------------------------------------------------------
        # 2. READ BYTES & HASH
        # -----------------------------------------------------------
        file_bytes = await file.read()
        file_size = len(file_bytes)
        file_hash = postgres_manager.compute_hash(file_bytes)

        # -----------------------------------------------------------
        # 3. DUPLICATE CHECK
        # -----------------------------------------------------------
        existing = postgres_manager.find_by_hash(file_hash)

        if existing:
            logger.info(f"[UPLOAD] Duplicate detected → {existing['doc_id']}")

            return DocumentUploadResponse(
                doc_id=str(existing["doc_id"]),
                name=existing["filename"],
                uploadedAt=existing["uploaded_at"].isoformat(),
                size=f"{existing['size_bytes']/1024:.1f} KB",
                page_count=existing["page_count"],
                is_first_time=False,
                previous_summary=doc_session_manager.get_document_summary(existing["doc_id"]),
                has_tables=existing["has_tables"],
                has_charts=existing["has_charts"]
            )

        # -----------------------------------------------------------
        # 4. WRITE TEMP FILE FOR GEMINI
        # -----------------------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name

        logger.info(f"[UPLOAD] Temp file created → {temp_path}")

        # -----------------------------------------------------------
        # 5. PROCESS WITH GEMINI
        # -----------------------------------------------------------
        processed = gemini_processor.process_document(temp_path)

        page_count = processed.get("page_count", 0)
        has_tables = len(processed.get("tables", [])) > 0
        has_charts = len(processed.get("charts", [])) > 0

        # -----------------------------------------------------------
        # 6. SAVE METADATA TO POSTGRES
        # -----------------------------------------------------------
        new_doc = postgres_manager.register_document(
            user_id=user_id,
            filename=file.filename,
            file_hash=file_hash,
            file_type=ext.lstrip("."),
            page_count=page_count,
            has_tables=has_tables,
            has_charts=has_charts,
            size_bytes=file_size
        )

        doc_id = new_doc["doc_id"]
        uploaded_at = new_doc["uploaded_at"]

        logger.info(f"[POSTGRES] Inserted new document → {doc_id}")

        # -----------------------------------------------------------
        # 7. STORE CHUNKS IN PINECONE
        # -----------------------------------------------------------
        gemini_processor.store_in_pinecone(
            processed=processed,
            doc_id=str(doc_id),
            filename=file.filename
        )

        logger.info(f"[PINECONE] Stored vectors for document → {doc_id}")

        # Clean temp file
        os.remove(temp_path)

        # -----------------------------------------------------------
        # 8. RESPONSE (no history yet)
        # -----------------------------------------------------------
        return DocumentUploadResponse(
            doc_id=str(doc_id),
            name=file.filename,
            uploadedAt=uploaded_at.isoformat(),
            size=f"{file_size/1024:.1f} KB",
            page_count=page_count,
            is_first_time=True,
            previous_summary=None,
            has_tables=has_tables,
            has_charts=has_charts,
        )

    except Exception as e:
        logger.error(f"[UPLOAD ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.delete("/delete-document/{doc_id}")
async def delete_document(doc_id: str):
    """
    Completely deletes a document:
    - remove Pinecone vectors
    - remove Postgres record
    - remove STM & LTM (Redis + Weaviate)
    """
    try:
        logger.info(f"[DELETE] Starting removal → doc_id={doc_id}")

        # -----------------------------------------------------------------
        # 1. Validate document
        # -----------------------------------------------------------------
        metadata = postgres_manager.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        # -----------------------------------------------------------------
        # 2. Delete Pinecone vectors
        # -----------------------------------------------------------------
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            index = pc.Index("trading-bot")

            index.delete(
                namespace="documents",
                filter={"doc_id": doc_id}
            )

            logger.info("[DELETE] Pinecone vectors removed")

        except Exception as e:
            logger.error(f"[DELETE] Pinecone cleanup failed: {e}")

        # -----------------------------------------------------------------
        # 3. Delete Postgres row
        # -----------------------------------------------------------------
        try:
            postgres_manager.delete_document(doc_id)
            logger.info("[DELETE] Postgres record removed")
        except Exception as e:
            logger.error(f"[DELETE] Postgres error: {e}")
            raise HTTPException(status_code=500, detail="Failed removing Postgres record")

        # -----------------------------------------------------------------
        # 4. Remove STM (Redis) & LTM (Weaviate)
        # -----------------------------------------------------------------
        try:
            # Redis STM
            document_workflow.memory_manager.delete_conversation(doc_id)

            # Weaviate LTM
            doc_session_manager.delete_document_history(doc_id)

            logger.info("[DELETE] STM + LTM removed")

        except Exception as e:
            logger.warning(f"[DELETE] Partial memory cleanup: {e}")

        # -----------------------------------------------------------------
        # 5. SUCCESS RESPONSE
        # -----------------------------------------------------------------
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[DELETE] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document-session/create")
async def create_document_session(req: DocumentSessionCreateRequest):
    """
    Creates a NEW document chat session:
    - session_id is newly generated (trade_doc_session_xxx)
    - mapped to doc_id using registry
    - initializes STM for doc_id
    - returns rolling summary if exists
    """
    try:
        doc_id = req.doc_id
        user_id = req.user_id

        logger.info(f"[DOC SESSION] Creating NEW session for doc_id={doc_id}")

        # Validate doc exists
        metadata = postgres_manager.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        # Create new session ID
        session_id = doc_session_registry.create_session(doc_id)

        # Initialize STM memory
        document_workflow.memory_manager.store_message(
            doc_id,
            "system",
            f"New document chat session created at {datetime.now().isoformat()}"
        )

        # Rolling summary
        summary = doc_session_manager.get_document_summary(doc_id)

        return {
            "session_id": session_id,
            "doc_id": doc_id,
            "summary": summary,
            "message": "New document chat session created",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[DOC SESSION] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/document-query", response_model=QueryResponse)
async def document_query(request: DocumentQueryRequest):
    try:
        logger.info(f"[DOC QUERY] Session: {request.session_id}")
        logger.info(f"[DOC QUERY] Q: {request.question[:80]}...")

        # Resolve session → doc_id
        mapped_doc_id = doc_session_registry.get_doc_id(request.session_id)
        if not mapped_doc_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid session_id: no document mapped"
            )

        doc_id = mapped_doc_id
        request.doc_id = doc_id

        # Validate doc exists
        doc_metadata = postgres_manager.get_document_metadata(doc_id)
        if not doc_metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        # Store STM message
        document_workflow.memory_manager.store_message(
            doc_id,
            "user",
            request.question
        )

        
        intent = classifier.classify(request.question)
        logger.info(f"[INTENT] Classified intent → {intent}")

        
        routed_msg = document_workflow.route_intent(
            intent=intent,
            question=request.question,
            doc_id=doc_id,
            user_id=request.user_id
        )

        # If the router handled it → NO RAG NEEDED
        if routed_msg is not None:
            answer = routed_msg.content

            # Still store Q&A in LTM for continuity
            doc_session_manager.store_document_qa(
                doc_id=doc_id,
                user_id=request.user_id,
                question=request.question,
                answer=answer,
                sources=[]
            )

            summary = doc_session_manager.get_document_summary(doc_id)

            return QueryResponse(
                answer=answer,
                session_id=request.session_id,
                message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now().isoformat(),
                sources=[],
                doc_summary=summary,
                query_type=intent,
                is_fast_path=True
            )

        
        initial_state = {
            "messages": [HumanMessage(content=request.question)],
            "doc_id": doc_id,
            "user_id": request.user_id,
            "retrieved_chunks": [],
            "sources": [],
            "gemini_analysis": "",
            "summary": ""
        }

        result = document_graph.invoke(initial_state)

        final_msgs = result.get("messages", [])
        answer = next(
            (m.content for m in reversed(final_msgs) if hasattr(m, "content")),
            "I couldn't find an answer in the document."
        )

        sources = result.get("sources", [])

        doc_session_manager.store_document_qa(
            doc_id=doc_id,
            user_id=request.user_id,
            question=request.question,
            answer=answer,
            sources=sources
        )

        summary = doc_session_manager.get_document_summary(doc_id)

        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
            timestamp=datetime.now().isoformat(),
            sources=sources,
            doc_summary=summary,
            query_type="document",
            is_fast_path=False
        )

    except Exception as e:
        logger.error(f"[DOC QUERY] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
@app.get("/get-document-chat-preview/{doc_id}", response_model=DocumentChatPreviewResponse)
async def get_document_chat_preview(doc_id: str):
    ""
    Returns:
    - doc metadata
    - summary
    - last 3 Q&A
    ""
    try:
        logger.info(f"[PREVIEW] doc_id={doc_id}")

        metadata = postgres_manager.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        summary = doc_session_manager.get_document_summary(doc_id)
        last_qa_raw = doc_session_manager.get_last_document_qa(doc_id, limit=3)

        # Convert to model list
        last_qa = [
            DocumentPreviewQA(
                question=item["question"],
                answer=item["answer"]
            )
            for item in last_qa_raw
        ]

        return DocumentChatPreviewResponse(
            doc_id=doc_id,
            filename=metadata["filename"],
            uploadedAt=metadata["uploaded_at"].isoformat(),
            summary=summary,
            last_qa=last_qa,
            page_count=metadata["page_count"],
            has_tables=metadata["has_tables"],
            has_charts=metadata["has_charts"]
        )

    except Exception as e:
        logger.error(f"[PREVIEW] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""

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
        USER_ID = "static_test_user"
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
            "user_id": USER_ID,
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

@app.get("/get-user-documents", response_model=UserDocumentsResponse)
async def get_user_documents(user_id: str = "static_test_user"):
    try:
        logger.info(f"[GET DOCS] Fetching docs for user: {user_id}")

        rows = postgres_manager.get_user_documents(user_id)
        documents = []

        for row in rows:
            documents.append(
                UserDocument(
                    id=str(row["doc_id"]),
                    name=row["filename"],
                    ploadedAt=row["uploaded_at"].isoformat(),
                    size=f"{row['size_bytes']/1024:.1f} KB",
                    page_count=row["page_count"],
                    has_tables=row["has_tables"],
                    has_charts=row["has_charts"],
                    summary=doc_session_manager.get_document_summary(str(row["doc_id"]))
                )
            )

        return UserDocumentsResponse(
            documents=documents,
            count=len(documents)
        )

    except Exception as e:
        logger.error(f"[GET DOCS] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-document-summary/{doc_id}", response_model=DocumentHistoryResponse)
async def get_document_summary(doc_id: str):
    try:
        logger.info(f"[GET SUMMARY] doc_id={doc_id}")

        metadata = postgres_manager.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        summary = doc_session_manager.get_document_summary(doc_id) or ""
        qa_count = doc_session_manager.get_document_qa_count(doc_id) 
        last_raw = doc_session_manager.get_last_document_qa(doc_id, limit=3) or []

        last_qa = [
            DocumentHistoryQA(
                question=item["question"],
                answer=item["answer"]
            )
            for item in last_raw
        ]

        return DocumentHistoryResponse(
            doc_id=doc_id,
            filename=metadata["filename"],
            first_time=(qa_count == 0),
            summary=summary,
            qa_count=qa_count,
            last_qa=last_qa
        )

    except Exception as e:
        logger.error(f"[GET SUMMARY] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-history/{doc_id}")
async def document_history(doc_id: str):
    """
    Unified endpoint for full document conversation history.
    Replaces:
    - /get-document-summary/{doc_id}
    - /get-document-chat-preview/{doc_id}
    
    Returns:
    - summary (LTM)
    - last 3 Q&A
    - qa_count
    - first_time
    - document metadata (Postgres)
    """
    try:
        logger.info(f"[DOC HISTORY] Fetching history for doc_id={doc_id}")

        # 1. Validate document exists in Postgres
        metadata = postgres_manager.get_document_metadata(doc_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Document not found")

        # 2. Fetch summary (Weaviate LTM)
        summary = doc_session_manager.get_document_summary(doc_id)

        # 3. Fetch last 3 interactions
        last_qa = doc_session_manager.get_last_document_qa(doc_id, limit=3)

        # 4. Total Q&A count
        qa_count = doc_session_manager.get_document_qa_count(doc_id)

        # 5. First-time flag
        first_time = (qa_count == 0)

        # 6. Return final unified response
        return {
            "doc_id": doc_id,
            "filename": metadata["filename"],
            "uploadedAt": metadata["uploaded_at"].isoformat(),
            "size": f"{metadata['size_bytes']/1024:.1f} KB",
            "page_count": metadata["page_count"],
            "has_tables": metadata["has_tables"],
            "has_charts": metadata["has_charts"],

            "summary": summary or "No previous conversations",
            "last_qa": last_qa,
            "qa_count": qa_count,
            "first_time": first_time
        }

    except Exception as e:
        logger.error(f"[DOC HISTORY] Error: {e}")
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