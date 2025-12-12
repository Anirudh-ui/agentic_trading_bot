"""
FastAPI Application - Final Production Version (Intent-First Routing)
- Gemini 2.5 Flash multimodal ingestion
- Groq conversational RAG
- Pre-graph intent classification
- S2 document summary (Gemini full-document summarizer)
- C1 conversation summary
- C2 chat history
- Q1 document query (graph)
"""

###############################################################
# IMPORTS
###############################################################
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
from utils.model_loaders import ModelLoader
from langchain_pinecone import PineconeVectorStore

from vertexai.generative_models import GenerativeModel
# Data models
from data_models.models import (
    DocumentUploadResponse,
    QueryResponse,
    DocumentSessionCreateRequest,
    DocumentHistoryQA,
    DocumentHistoryResponse,
    UserDocumentsResponse,
    UserDocument,
    InternetQueryRequest
)

from pinecone import Pinecone
from langchain_core.messages import HumanMessage, AIMessage

###############################################################
# FASTAPI INIT
###############################################################
app = FastAPI(
    title="Trading Bot - Multimodal System (FINAL)",
    version="3.0.0",
    description="Two-LLM architecture with intent classification and full document routing"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

###############################################################
# GLOBAL INITIALIZATION
###############################################################

logger.info("============================================")
logger.info("🚀 Starting Trading Bot (Final Version)")
logger.info("============================================")

# Intent classifier (LLM-powered)
intent_classifier = DocumentIntentClassifier()
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
#postgres manager
try:
    postgres_manager = PostgresManager()
    logger.info("[STARTUP] postgres processor initialized")
except Exception as e:
    logger.error(f"[STARTUP] postgres processor failed: {e}")
    raise
# Document session manager

try:
    doc_session_manager = DocumentSessionManager(
       
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


###############################################################
# ROUTES
###############################################################

@app.get("/")
def root():
    return {"status": "online", "message": "Trading Bot Final Version"}


#######################################################################
# DOCUMENT UPLOAD
#######################################################################
@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        logger.info(f"[UPLOAD] {file.filename}")

        user_id = "static_test_user"
        ext = os.path.splitext(file.filename)[1].lower()

        allowed = [".pdf", ".docx", ".txt"]
        if ext not in allowed:
            raise HTTPException(status_code=400, detail="Unsupported file type.")

        # read bytes
        file_bytes = await file.read()
        file_hash = postgres_manager.compute_hash(file_bytes)

        # DUPLICATE CHECK
        existing = postgres_manager.find_by_hash(file_hash)
        if existing:
            logger.info("[UPLOAD] Duplicate detected.")
            summary = doc_session_manager.get_document_summary(existing["doc_id"])

            return DocumentUploadResponse(
                doc_id=str(existing["doc_id"]),
                name=existing["filename"],
                uploadedAt=existing["uploaded_at"].isoformat(),
                size=f"{existing['size_bytes']/1024:.1f} KB",
                page_count=existing["page_count"],
                has_tables=existing["has_tables"],
                has_charts=existing["has_charts"],
                is_first_time=False,
                previous_summary=summary
            )

        # TEMP FILE
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        processed = gemini_processor.process_document(temp_path)

        page_count = processed["page_count"]
        has_tables = len(processed["tables"]) > 0
        has_charts = len(processed["charts"]) > 0
        size_bytes = len(file_bytes)

        # REGISTER DOC
        new_doc = postgres_manager.register_document(
            user_id=user_id,
            filename=file.filename,
            file_hash=file_hash,
            file_type=ext.replace(".", ""),
            page_count=page_count,
            has_tables=has_tables,
            has_charts=has_charts,
            size_bytes=size_bytes
        )

        doc_id = new_doc["doc_id"]

        # STORE IN PINECONE
        gemini_processor.store_in_pinecone(processed, str(doc_id), file.filename)

        os.remove(temp_path)

        return DocumentUploadResponse(
            doc_id=str(doc_id),
            name=file.filename,
            uploadedAt=new_doc["uploaded_at"].isoformat(),
            size=f"{size_bytes/1024:.1f} KB",
            page_count=page_count,
            is_first_time=True,
            previous_summary=None,
            has_tables=has_tables,
            has_charts=has_charts
        )

    except Exception as e:
        logger.error(f"[UPLOAD ERROR] {e}")
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

        summary = doc_session_manager.get_document_summary(doc_id)
        qa_count = doc_session_manager.get_document_qa_count(doc_id)
        last_raw = doc_session_manager.get_last_document_qa(doc_id, limit=3)

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
#######################################################################
# CREATE DOCUMENT SESSION
#######################################################################
@app.post("/document-session/create")
async def create_document_session(req: DocumentSessionCreateRequest):
    try:
        doc_id = req.doc_id
        meta = postgres_manager.get_document_metadata(doc_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Document not found")

        session_id = doc_session_registry.create_session(doc_id)

        # write STM
        document_workflow.memory_manager.store_message(
            doc_id,
            "system",
            f"New document session created {datetime.now().isoformat()}"
        )

        summary = doc_session_manager.get_document_summary(doc_id)

        return {
            "session_id": session_id,
            "doc_id": doc_id,
            "summary": summary,
            "message": "Session created",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[SESSION CREATE ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


#######################################################################
# S2 — FULL DOCUMENT SUMMARY (Gemini full-document summary)
#######################################################################
def generate_full_document_summary(doc_id: str):
    """
    Uses Gemini to summarize ALL chunks from Pinecone.
    """
    try:
        # load embeddings & pinecone
        ml = ModelLoader()
        embed = ml.load_embeddings_vertex()

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("trading-bot")

        vs = PineconeVectorStore(
            index=index,
            embedding=embed,
            namespace="documents"
        )

        # fetch all chunks
        results = vs.similarity_search(query="*", k=1000, filter={"doc_id": doc_id})

        text = "\n\n".join(doc.page_content for doc in results)

        gemini = GenerativeModel("gemini-2.5-flash")
        prompt = f"""
Summarize the entire following document in under 200 words.
No hallucination.
Only use text provided.

DOCUMENT CONTENT:
{text}
"""

        resp = gemini.generate_content(prompt)
        return resp.text.strip()

    except Exception as e:
        logger.error(f"[FULL SUMMARY ERROR] {e}")
        return "Summary unavailable."


#######################################################################
# DOCUMENT QUERY (INTENT FIRST → THEN ROUTE)
#######################################################################
@app.post("/document-query", response_model=QueryResponse)
async def document_query(request: DocumentQueryRequest):
    try:
        logger.info(f"[DOC QUERY] Q={request.question[:80]}")

        # Resolve session → doc_id
        doc_id = doc_session_registry.get_doc_id(request.session_id)
        if not doc_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")

        # Store STM user message
        document_workflow.memory_manager.store_message(
            doc_id, "user", request.question
        )

        # STEP 1 — INTENT CLASSIFICATION
        intent = intent_classifier.classify(request.question)
        logger.info(f"[INTENT] → {intent}")

        ###################################################################
        # INTENT: DOCUMENT_SUMMARY (S2)
        ###################################################################
        if intent == "DOCUMENT_SUMMARY":
            summary = generate_full_document_summary(doc_id)

            # store in LTM
            doc_session_manager.store_summary(doc_id, summary, topics=["document_summary"])

            return QueryResponse(
                answer=summary,
                session_id=request.session_id,
                message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now().isoformat(),
                sources=None,
                doc_summary=summary,
                query_type="document_summary",
                is_fast_path=True
            )

        ###################################################################
        # INTENT: CONVERSATION_SUMMARY (C1)
        ###################################################################
        if intent == "CONVERSATION_SUMMARY":
            summary = doc_session_manager.get_document_summary(doc_id)
            if not summary:
                summary = "No conversation history available."

            return QueryResponse(
                answer=summary,
                session_id=request.session_id,
                message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now().isoformat(),
                sources=None,
                doc_summary=summary,
                query_type="conversation_summary",
                is_fast_path=True
            )

        ###################################################################
        # INTENT: CONVERSATION_HISTORY (C2)
        ###################################################################
        if intent == "CONVERSATION_HISTORY":
            history = doc_session_manager.get_last_document_qa(doc_id, limit=20)

            history_text = "\n\n".join(
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in history
            )

            if not history:
                history_text = "No previous messages."

            return QueryResponse(
                answer=history_text,
                session_id=request.session_id,
                message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
                timestamp=datetime.now().isoformat(),
                sources=None,
                doc_summary=None,
                query_type="conversation_history",
                is_fast_path=True
            )

        ###################################################################
        # INTENT: DOCUMENT_QUERY (Q1) — Use GRAPH
        ###################################################################
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

        final_messages = result.get("messages", [])
        answer = next(
            (m.content for m in reversed(final_messages) if hasattr(m, "content")),
            "I could not find an answer."
        )

        sources = result.get("sources", [])
        summary = doc_session_manager.get_document_summary(doc_id)

        # Store Q&A LTM
        doc_session_manager.store_document_qa(
            doc_id, request.user_id, request.question, answer, sources
        )

        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
            timestamp=datetime.now().isoformat(),
            sources=sources,
            doc_summary=summary,
            query_type="document_query",
            is_fast_path=False
        )

    except Exception as e:
        logger.error(f"[DOC QUERY ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


#######################################################################
# UNIFIED DOCUMENT HISTORY
#######################################################################
@app.get("/document-history/{doc_id}", response_model=DocumentHistoryResponse)
async def document_history(doc_id: str):
    try:
        meta = postgres_manager.get_document_metadata(doc_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Document not found")

        summary = doc_session_manager.get_document_summary(doc_id)
        last_qa = doc_session_manager.get_last_document_qa(doc_id, limit=3)
        qa_count = doc_session_manager.get_document_qa_count(doc_id)

        last_qa_models = [
            DocumentHistoryQA(question=i["question"], answer=i["answer"]) for i in last_qa
        ]

        return DocumentHistoryResponse(
            doc_id=doc_id,
            filename=meta["filename"],
            first_time=(qa_count == 0),
            summary=summary,
            last_qa=last_qa_models,
            qa_count=qa_count
        )
    except Exception as e:
        logger.error(f"[HISTORY ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


#######################################################################
# INTERNET ENDPOINT (unchanged)
#######################################################################
@app.post("/internet-query", response_model=QueryResponse)
async def internet_query(request: InternetQueryRequest):
    try:
        initial_state = {
            "messages": [HumanMessage(content=request.question)],
            "query_type": "unknown",
            "session_id": request.session_id,
            "user_id": "static_user",
            "is_fast_path": False,
            "tickers": [],
            "needs_tools": True
        }

        result = internet_graph.invoke(initial_state)

        final_messages = result.get("messages", [])
        answer = next(
            (m.content for m in reversed(final_messages) if hasattr(m, "content")),
            "I couldn't process that."
        )

        return QueryResponse(
            answer=answer,
            session_id=request.session_id,
            message_id=f"msg_{int(datetime.now().timestamp()*1000)}",
            timestamp=datetime.now().isoformat(),
            sources=None,
            query_type=result.get("query_type", "general"),
            is_fast_path=result.get("is_fast_path", False)
        )

    except Exception as e:
        logger.error(f"[INET ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


###############################################################
# STARTUP / SHUTDOWN
###############################################################

@app.on_event("startup")
def startup_event():
    logger.info("API Startup Complete")


@app.on_event("shutdown")
def shutdown_event():
    document_workflow.close()
    logger.info("Shutdown complete.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )