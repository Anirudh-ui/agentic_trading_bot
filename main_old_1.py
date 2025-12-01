from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from data_models.models import QueryModel, SessionIDModel
from agent.workflow_with_memory import GraphBuilder
from langchain_core.messages import HumanMessage
from data_ingestion.ingestion_pipeline import DataIngestion
from exception.exceptions import TradingBotException
import sys
from typing import List
import os
import json
from fastapi.responses import JSONResponse
app = FastAPI(title="Stock Market Agentic Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize graph builder with memory
graph_builder = GraphBuilder()
graph_builder.build()
graph = graph_builder.get_graph()

# Initialize document processor
doc_processor = DataIngestion()


@app.get("/")
async def root():
    return {
        "message": "Stock Market Agentic Chatbot API",
        "endpoints": {
            "/query": "POST - Send a query to the chatbot",
            "/upload": "POST - Upload documents for knowledge base",
            "/clear-session": "POST - Clear session memory",
            "/session-info": "POST - Get session information",
            "/user-profile": "GET - Get user profile",
            "/health": "GET - Health check"
        }
    }

# In main.py, before the /query endpoint

def format_final_answer(answer_string: str) -> str:
    """
    Parses the structured answer (if present) from tools or returns the plain answer, 
    and formats it for user-friendly display (with citations).
    """
    try:
        # 1. Attempt to parse the answer as a structured JSON object (from search tool)
        data = json.loads(answer_string)
        
        # Check if it has the required fields from a structured search tool response
        if all(key in data for key in ["answer", "results"]):
            
            final_response = data["answer"].strip()
            
            # 2. Append formatted sources (citations)
            if data["results"]:
                final_response += "\n\n**Sources:**\n"
                for i, result in enumerate(data["results"]):
                    # Use a basic markdown link format
                    final_response += f"* [{result['title']}]({result['url']})\n"
                    
            return final_response
            
    except json.JSONDecodeError:
        # If it's not a JSON string, it's a plain text response (from general chatbot/memory)
        pass
    except Exception as e:
        # Handle cases where JSON is present but structure is unexpected
        print(f"[FORMATTING ERROR] Failed to format structured response: {e}")
        
    # If the above fails, just return the original string (clean it up slightly)
    return answer_string.strip()
@app.post("/query")
async def query_bot(request: QueryModel):
    """
    Process user query with session-based memory
    
    Args:
        request: QueryModel containing question and session_id
    
    Returns:
        dict with answer and session information
    """
    try:
        question = request.question
        session_id = request.session_id
        
        # Extract user_id from session_id if it follows pattern "trade_app_<uuid>"
        # In production, you'd authenticate and get real user_id
        #user_id = session_id.replace("trade_app_", "") if session_id.startswith("trade_app_") else "anonymous"
        user_id = "static_test_user"
        print(f"\n[REQUEST] Session: {session_id}")
        print(f"[REQUEST] User ID: {user_id}")
        print(f"[REQUEST] Question: {question}")
        
        # Store user message in Redis immediately
        graph_builder.memory_manager.store_message(session_id, "user", question)
        
        # Get conversation history from Redis for context
        short_term_history = graph_builder.memory_manager.get_short_term_memory(session_id, limit=10)
        
        # Prepare initial state with memory context
        initial_state = {
            "messages": short_term_history + [HumanMessage(content=question)],
            "query_type": "unknown",
            "relevance_score": 0.0,
            "needs_correction": False,
            "summary": "",
            "session_id": session_id,
            "user_id": user_id
        }
        
        # Invoke graph
        result = graph.invoke(initial_state)
        
        # Extract final answer
        final_messages = result.get("messages", [])
        answer = "I couldn't process that request."
        
        for msg in reversed(final_messages):
            if hasattr(msg, 'content') and msg.content:
                answer = msg.content
                break
        final_answer = format_final_answer(answer)
        # Get session metadata
        session_metadata = graph_builder.memory_manager.redis_manager.get_session_metadata(session_id)
        
        response = {
            "answer": final_answer,
            "session_id": session_id,
            "user_id": user_id,
            "message_count": session_metadata.get('message_count', 0) if session_metadata else 0,
            "query_type": result.get("query_type", "unknown")
        }
        
        print(f"[RESPONSE] Answer: {answer[:100]}...")
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Query processing failed: {str(e)}")
        raise TradingBotException(e, sys)


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        ingestion = DataIngestion()
        ingestion.run_pipeline(files)
        return {"message": "Files successfully processed and stored."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/clear-session")
async def clear_session(request: SessionIDModel):
    """
    Clear session memory from Redis
    
    Args:
        request: SessionIDModel with session_id
    
    Returns:
        dict with status
    """
    try:
        session_id = request.session_id
        graph_builder.memory_manager.redis_manager.clear_session(session_id)
        
        return {
            "message": f"Session {session_id} cleared successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.post("/session-info")
async def get_session_info(request: SessionIDModel):
    """
    Get session information and metadata
    
    Args:
        request: SessionIDModel with session_id
    
    Returns:
        dict with session metadata and recent messages
    """
    try:
        session_id = request.session_id
        
        # Get metadata from Redis
        metadata = graph_builder.memory_manager.redis_manager.get_session_metadata(session_id)
        
        # Get recent messages
        messages = graph_builder.memory_manager.redis_manager.get_messages(session_id, limit=5)
        
        return {
            "session_id": session_id,
            "metadata": metadata,
            "recent_messages": messages
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.get("/user-profile/{user_id}")
async def get_user_profile(user_id: str):
    """
    Get user profile from Weaviate long-term memory
    
    Args:
        user_id: User identifier
    
    Returns:
        dict with user profile data
    """
    try:
        profile = graph_builder.memory_manager.weaviate_manager.get_user_profile(user_id)
        
        if not profile:
            return {
                "message": "No profile found for this user",
                "user_id": user_id
            }
        
        return {
            "user_id": user_id,
            "profile": profile
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.get("/conversation-history/{user_id}")
async def get_conversation_history(user_id: str, limit: int = 10):
    """
    Get user's conversation history from Weaviate
    
    Args:
        user_id: User identifier
        limit: Number of conversations to retrieve
    
    Returns:
        dict with conversation summaries
    """
    try:
        history = graph_builder.memory_manager.weaviate_manager.get_user_conversation_history(
            user_id, limit
        )
        
        return {
            "user_id": user_id,
            "conversation_count": len(history),
            "conversations": history
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.get("/active-sessions")
async def get_active_sessions():
    """
    Get list of active sessions in Redis
    
    Returns:
        dict with active session IDs
    """
    try:
        sessions = graph_builder.memory_manager.redis_manager.get_active_sessions()
        
        return {
            "active_sessions": len(sessions),
            "session_ids": sessions
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.post("/archive-session")
async def archive_session(request: SessionIDModel):
    """
    Manually archive a session to Weaviate
    
    Args:
        request: SessionIDModel with session_id
    
    Returns:
        dict with status
    """
    try:
        session_id = request.session_id
        user_id = session_id.replace("trade_app_", "") if session_id.startswith("trade_app_") else "anonymous"
        
        # Get messages from Redis
        messages = graph_builder.memory_manager.redis_manager.get_messages(session_id)
        
        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this session")
        
        # Generate summary
        # (You could use the summary_chain here for a better summary)
        summary = f"Conversation with {len(messages)} messages"
        
        # Extract topics and preferences
        key_topics = ["trading", "stocks"]  # Simplified
        user_preferences = {}
        
        # Archive to Weaviate
        graph_builder.memory_manager.archive_conversation(
            session_id=session_id,
            summary=summary,
            key_topics=key_topics,
            user_preferences=user_preferences,
            user_id=user_id
        )
        
        return {
            "message": "Session archived successfully",
            "session_id": session_id,
            "message_count": len(messages)
        }
        
    except Exception as e:
        raise TradingBotException(e, sys)


@app.get("/health")
async def health_check():
    """
    Health check endpoint with memory system status
    
    Returns:
        dict with health status
    """
    try:
        # Check Redis connection
        redis_status = "healthy"
        try:
            graph_builder.memory_manager.redis_manager.redis_client.ping()
        except:
            redis_status = "unhealthy"
        
        # Check Weaviate connection
        weaviate_status = "healthy"
        try:
            graph_builder.memory_manager.weaviate_manager.client.is_ready()
        except:
            weaviate_status = "unhealthy"
        
        return {
            "status": "healthy" if redis_status == "healthy" and weaviate_status == "healthy" else "degraded",
            "redis": redis_status,
            "weaviate": weaviate_status,
            "cache_stats": graph_builder.get_cache_stats()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    print("[SHUTDOWN] Closing memory connections...")
    graph_builder.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
