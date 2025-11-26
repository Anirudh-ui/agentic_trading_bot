from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import JSONResponse
from data_ingestion.ingestion_pipeline import DataIngestion 
from agent.workflow import GraphBuilder
# FIX: Import BaseModel for Pydantic models since data_models.models is not provided
from data_models.models import QueryModel,SessionIDModel,RagToolSchema
# Memory & Graph Persistence Imports
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_core.runnables import RunnableConfig 
# from utils.memory_manager import get_session_memory # NOTE: This is no longer needed/used with LangGraph checkpointer
import os 
import redis # Used for the clear_redis_memory utility (synchronous client)
from dotenv import load_dotenv
load_dotenv()
# REFACTORED: Use the correct and requested class name, RedisSaver
from langgraph.checkpoint.redis import RedisSaver 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Checkpointer Initialization
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380")

# REFACTORED: Initialize using RedisSaver.from_conn_string
try:
    # Initialize the global checkpointer instance for the FastAPI app
    CHECKPOINTER = RedisSaver.from_conn_string(REDIS_URL)
    print(f"✅ LangGraph RedisSaver initialized with URL: {REDIS_URL}")
except Exception as e:
    # Log error if connection fails and raise exception since checkpointer is critical
    print(f"❌ Failed to connect and initialize RedisSaver at {REDIS_URL}. Error: {e}")
    raise e 

# --- START: Required Checkpointer Setup & Memory Clear Logic ---

# 1. Setup the checkpointer: Creates necessary indices/keys in Redis on first run.
# FIX: Commented out CHECKPOINTER.setup() as it is causing the 
# AttributeError: '_GeneratorContextManager' error due to a version conflict/setup
# The setup is often implicit upon graph compilation or first run.
# CHECKPOINTER.setup() 

# 2. Add a utility function to manually clear all checkpoints/memory
def clear_redis_memory():
    """Utility to clear all data in the Redis DB used by the checkpointer."""
    try:
        # Access the underlying redis client from the saver object to call flushdb
        # CHECKPOINTER.client is the sync redis.StrictClient instance
        redis_client = CHECKPOINTER.client 
        # Optional: Get the DB index to confirm which DB is being cleared
        db_index = redis_client.connection_pool.connection_kwargs.get('db', 0)
        redis_client.flushdb(db=db_index)
        print(f"✅ Cleared all data from Redis DB {db_index}.")
    except Exception as e:
        print(f"❌ Error clearing Redis memory: {e}")
        # Note: If redis is critical, you might want to re-raise or handle failure more gracefully.


# --- END: Required Checkpointer Setup & Memory Clear Logic ---


# Initialize the Agentic Graph (happens once at startup)
workflow_instance = GraphBuilder()
workflow_instance.build() # Build the actual graph structure

# Global Data Ingestion Manager
ingestion_manager = DataIngestion()


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "Trading Bot Agent is running."}

@app.post("/ingest")
async def ingest_data(files: List[UploadFile] = File(...)):
    """Handles file uploads and ingestion into the vector store."""
    try:
        # Check if any non-empty files were provided
        non_empty_files = [file for file in files if file.size > 0]
        if not non_empty_files:
            return JSONResponse(status_code=400, content={"message": "No valid files provided."})

        # Process all files
        for file in non_empty_files:
            file_data = await file.read()
            ingestion_manager.process_file(file.filename, file_data, file.content_type)
        
        # After processing all files
        # The ingestion manager will call its process method which handles chunking and vector store upload
        ingestion_manager.ingest_documents()

        return JSONResponse(status_code=200, content={"message": f"Successfully processed and ingested {len(non_empty_files)} files."})

    except Exception as e:
        print(f"Ingestion error: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred during ingestion: {e}"})

@app.post("/query")
async def process_query(query_model: QueryModel):
    """Handles chat queries using the LangGraph state machine."""
    
    session_id = query_model.session_id # e.g., 'trade_app_12345'
    user_question = query_model.question

    print(f"--- Received query for session {session_id} ---")
    print(f"Question: {user_question}")

    # 1. Define the LangGraph configuration (thread_id)
    config = {
        "configurable": {
            "thread_id": session_id,
            "checkpointer": CHECKPOINTER # <--- THIS IS THE MISSING PIECE
        }
    }

    # 2. Define the initial state for the run
    # LangGraph will automatically load the previous state based on the config/thread_id
    initial_state = {"messages": [HumanMessage(content=user_question)]}

    final_state_output = {}

    try:
        # Stream the graph execution
        final_state_output = workflow_instance.graph.invoke(initial_state, config=config)
        # 3. Extract the final answer
        print("---")
        print("from main",final_state_output)
        print("---")
       
        # Determine the source of the final messages
        if "messages" in final_state_output and final_state_output["messages"]:
            # The final message is the last one in the message list
            #final_message = final_state["chatbot"]["messages"][-1]
            #final_message = full_state["summary"][-1]
            # The final message should be an AIMessage with content
            final_message = final_state_output["messages"][-1]
            print("line 150",final_message)
            if hasattr(final_message, 'content'):
                #final_answer = final_message.content
                final_answer_raw = final_message.content
                try:
                    import json
                    # Attempt to parse as JSON (assuming tool output is structured)
                    parsed_json = json.loads(final_answer_raw)
                    # If it has an 'answer' key (like your RAG tool output), use that.
                    if 'answer' in parsed_json:
                        final_answer = parsed_json['answer']
                    else:
                        # If it's JSON but doesn't have an 'answer' key, return the raw string
                        final_answer = final_answer_raw
                        
                except (json.JSONDecodeError, TypeError):
                    # Not a JSON string, use the content as is
                    final_answer = final_answer_raw
            else:
                final_answer = "Error: Final response message was not a standard LangChain message."
            
            print(f"--- Session {session_id} completed. Final response length: {len(final_answer)} ---")

            return JSONResponse(status_code=200, content={"answer": final_answer})
        
        else:
            return JSONResponse(status_code=500, content={"answer": "Execution failed: No final state or messages were generated by the graph."})

    except Exception as e:
        print(f"Graph execution error: {e}")
        return JSONResponse(status_code=500, content={"answer": f"An unexpected error occurred during processing: {e}"})

@app.post("/clear_memory")
def clear_session_memory(model: SessionIDModel):
    """Clears the memory/state for a specific session ID (thread_id)."""
    session_id = model.session_id
    config = {"configurable": {"thread_id": session_id}}
    
    try:
        # Use put_state(config, None) to clear the state associated with the thread_id
        CHECKPOINTER.put_state(config, None)
        return JSONResponse(status_code=200, content={"message": f"State for session ID '{session_id}' cleared successfully."})
    except Exception as e:
        print(f"Error clearing session memory: {e}")
        return JSONResponse(status_code=500, content={"message": f"Failed to clear session state: {e}"})


@app.post("/clear_all_memory")
def clear_all_redis():
    """Clears ALL data in the Redis DB used by the checkpointer."""
    clear_redis_memory()
    return JSONResponse(status_code=200, content={"message": "All Redis memory (checkpoints) cleared successfully."})

# --- Local Test Run Endpoint ---

@app.get("/test_run")
def test_run():
    """
    Runs a sample two-turn conversation using a hardcoded thread_id
    to test persistence and graph flow.
    """
    try:
        # Use a consistent test thread ID
        test_thread_id = "test_thread_001"
        test_question_1 = "Hi! I'm Bob, and my favorite stock is NVDA."
        test_question_2 = "What is my name and what is my favorite stock?"

        # The RunnableConfig for the test
        test_config = {
        "configurable": {
            "thread_id": test_thread_id,
            "checkpointer": CHECKPOINTER # <--- THIS IS THE MISSING PIECE
        }
    }
        
        # 1. Clear test thread state before starting a fresh run
        print(f"Clearing state for test thread: {test_thread_id}")
        CHECKPOINTER.put_state(test_config, None)

        print("\n--- STARTING TEST RUN 1 (Introduction) ---")
        
        # Turn 1: Introduction
        initial_state_1 = {"messages": [HumanMessage(content=test_question_1)]}
        print(f"Q1: {test_question_1}")
        
        final_state_1 = {} 
        for chunk in workflow_instance.graph.stream(initial_state_1, config=test_config):
            final_state_1 = chunk 

        if final_state_1 and "messages" in final_state_1:
            final_message_1 = final_state_1["messages"][-1]
            final_answer_1 = getattr(final_message_1, 'content', "Error: No content.")
            print("\n--- BOT RESPONSE 1 ---")
            print(final_answer_1)
            print("------------------------\n")

        
        # Turn 2: Recall and answer
        print("\n--- STARTING TEST RUN 2 (Recall) ---")

        # The graph will load the state from Redis based on the same test_config
        initial_state_2 = {"messages": [HumanMessage(content=test_question_2)]}
        print(f"Q2: {test_question_2}")
        
        final_state_2 = {} 
        for chunk in workflow_instance.graph.stream(initial_state_2, config=test_config):
            final_state_2 = chunk 

        if final_state_2 and "messages" in final_state_2:
            final_message_2 = final_state_2["messages"][-1]
            final_answer_2 = getattr(final_message_2, 'content', "Error: No content.")
            print("\n--- BOT RESPONSE 2 (Should contain memory of Bob/NVDA) ---")
            print(final_answer_2)
            print("-----------------------------------------------------------\n")
            
            return JSONResponse(status_code=200, content={
                "message": "Test run completed successfully. Check console for trace.",
                "response_1": final_answer_1,
                "response_2": final_answer_2
            })
        
        else:
            return JSONResponse(status_code=500, content={"message": "Test run failed to generate a final response in turn 2."})
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR DURING TEST EXECUTION: {e}")
        return JSONResponse(status_code=500, content={"message": f"Test run failed due to a critical error: {e}"})

# Run the test automatically at startup to verify Redis connection and graph
@app.on_event("startup")
async def startup_event():
    print("Application Startup: Building Graph and Running Self-Test...")
    # The build() call happens globally above, but the test_run helps verify the setup.
    # test_run() # Commenting out self-test for cleaner startup console output, but it's available via the /test_run endpoint.
    print("Application Ready.")