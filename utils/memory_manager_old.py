import os
from dotenv import load_dotenv # Assuming you use dotenv for REDIS_URL
from langchain_community.chat_message_histories import RedisChatMessageHistory
from custom_logging.my_logger import logger 
from utils.model_loaders import ModelLoader
from langchain_classic.memory import ConversationSummaryBufferMemory
# Only load_dotenv once if possible, but harmless to do again
load_dotenv() 
# Default to a local Redis for development, use environment variable for production
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0") 

# Max token limit changed to 3000 to increase summarization prune logic
def get_session_memory(agent_bot_session_id: str, max_token_limit: int = 3000) -> ConversationSummaryBufferMemory:
    """
    Initializes a two-tier memory system (Redis persistence + Summarization logic).
    The max_token_limit controls when the LLM summarizes the history.
    """
    try:
        # L1: Persistent Store (Redis) - stores all raw messages
        # Note: This is separate from the LangGraph Checkpointer but uses the same Redis instance.
        history = RedisChatMessageHistory(
            session_id=agent_bot_session_id, 
            url=REDIS_URL
        )

        # L2: Summarization/Prune Logic - summarizes old messages using an LLM
        llm_for_summary = ModelLoader().load_llm() 

        memory = ConversationSummaryBufferMemory(
            llm=llm_for_summary,
            chat_memory=history,
            max_token_limit=max_token_limit, # Increased to 3000 as requested
            return_messages=True,
            memory_key="chat_history" 
        )
        
        logger.info(f"Initialized summary buffer memory with {max_token_limit} tokens for session: {agent_bot_session_id}")
        return memory

    except Exception as e:
        logger.error(f"Failed to initialize Redis memory for session {agent_bot_session_id}. Error: {e}")
        # Fallback to a simple in-memory buffer 
        from langchain.memory import ConversationBufferMemory
        logger.warning("Using non-persistent ConversationBufferMemory as fallback.")
        return ConversationBufferMemory(return_messages=True, memory_key="chat_history")