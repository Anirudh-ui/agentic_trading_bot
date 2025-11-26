from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *
from toolkit.smart_router import SmartToolRouter
from utils.memory_manager import HybridMemoryManager
import re
from functools import lru_cache
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
import os
import hashlib

class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str
    relevance_score: float
    needs_correction: bool
    summary: str
    session_id: str  # Added for memory management
    user_id: str  # Added for user identification


class ResponseCache:
    """Simple in-memory cache with TTL support"""
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def _generate_key(self, messages):
        message_text = " ".join([msg.content if hasattr(msg, 'content') else str(msg) for msg in messages[-3:]])
        return hashlib.md5(message_text.encode()).hexdigest()
    
    def get(self, messages):
        key = self._generate_key(messages)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, messages, response):
        key = self._generate_key(messages)
        self.cache[key] = (response, datetime.now())


class FastPathRouter:
    """Router for handling generic messages without LLM"""
    
    GREETING_PATTERNS = [
        r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
        r'^(yo|sup|what\'s up|howdy)',
    ]
    
    FAREWELL_PATTERNS = [r'\b(bye|goodbye|see you|farewell|take care)\b']
    THANKS_PATTERNS = [r'\b(thanks|thank you|thx|appreciated)\b']
    
    GENERIC_RESPONSES = {
        'greeting': "Hello! I'm your trading assistant. I can help you with:\n- Real-time stock prices and market data\n- Latest financial news\n- Trading strategies from our knowledge base\n\nWhat would you like to know?",
        'farewell': "Goodbye! Feel free to return if you have more questions about trading or the stock market.",
        'thanks': "You're welcome! Let me know if you need anything else related to trading or finance.",
    }
    
    @classmethod
    def is_generic_message(cls, message_content):
        content_lower = message_content.lower().strip()
        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return 'greeting'
        for pattern in cls.FAREWELL_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return 'farewell'
        for pattern in cls.THANKS_PATTERNS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return 'thanks'
        return None
    
    @classmethod
    def get_fast_response(cls, message_type):
        return cls.GENERIC_RESPONSES.get(message_type, None)


class GraphBuilder:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        
        # Initialize smart router for tool selection
        self.smart_router = SmartToolRouter()
        
        # Initialize hybrid memory manager
        self.memory_manager = HybridMemoryManager(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6380)),
            weaviate_url=os.getenv('WEAVIATE_URL', 'http://localhost:8080'),
            weaviate_api_key=os.getenv('WEAVIATE_API_KEY'),
            session_ttl_hours=24
        )
        
        # Summary prompt for progressive summarization
        summary_prompt = PromptTemplate.from_template(
            """
            Progressively summarize the given conversation for a financial trading assistant. 
            Keep the summary concise, under 100 words, and capture:
            - User's name (if mentioned)
            - Preferred stocks and financial instruments
            - Trading goals and strategies discussed
            - Key decisions or insights
            
            Current Summary: {current_summary}
            New Messages: {new_messages}
            
            Return ONLY the updated summary text.
            """
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        
        # Tools with clear purposes
        self.tools = [retriever_tool, yahoo_finance_tool, tavilytool]
        
        # System prompt for better tool calling
        system_prompt = """You are the **Indian Stock Market Expert**, a sophisticated, agentic trading assistant. Your core function is to provide accurate, context-aware information.

**RULES FOR TOOL USAGE & FALLBACKS (CRITICAL):**
1.  **RAG/Knowledge Base (Pinecone):** Always check the knowledge base first for trading concepts, strategies, or known document content.
2.  **Web Search (Tavily):**
    * **MUST** be used for any general, time-sensitive, or broad market questions (e.g., 'features of Indian stock market,' 'latest market news').
    * **MUST** be used if the RAG tool fails to find relevant information.
    * **MUST** be used to find specific stock **ticker symbols** if the user asks a general question like 'important Indian stocks' before attempting to use the Yahoo Finance tool.
3.  **Stock Data (Yahoo Finance):** Only use this tool if the user provides a specific, valid **stock ticker symbol** (e.g., RIL.NS, HDFC, INFY.NS). If a ticker is missing, use Web Search first.

**RULES FOR MEMORY INTEGRATION:**
* **User Profile:** The current user's profile is: {profile_context}
* **Long-Term Context:** Previous sessions summary: {long_term_context}
* **Mandate:** Always use the profile (name, goals, preferences) and long-term context to personalize your answer, even if subtly.
* **Short-Term History:** The recent conversation history is provided below. Use it to maintain coherence.

**CONVERSATION HISTORY:**
{redis_history}

**FINAL INSTRUCTION:** Analyze the history, context, and query. If a tool is necessary, call it immediately. If not, generate a concise, informed response.
"""

        llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.llm_with_tools = llm_with_tools
        self.system_message = SystemMessage(content=system_prompt)
        self.graph = None
        
        # Caching
        self.response_cache = ResponseCache(ttl_seconds=1800)
        self.fast_path_router = FastPathRouter()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _invoke_llm_with_retry(self, messages):
        """Invoke LLM with retry logic for rate limit handling"""
        try:
            return self.llm_with_tools.invoke(messages)
        except Exception as e:
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'quota' in error_msg or '429' in error_msg:
                print(f"Rate limit hit, retrying... Error: {e}")
                raise
            else:
                raise
    
    def _query_classifier_node(self, state: State):
        """Classify query type to route to appropriate tools"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, 'content'):
            return state
        
        query = last_message.content
        query_type = self.smart_router.classify_query(query)
        
        print(f"[ROUTER] Query classified as: {query_type}")
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _chatbot_node(self, state: State):
        """Enhanced chatbot node with Redis short-term memory integration"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "anonymous")
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, 'content'):
            return state

        message_content = last_message.content
        query_type = state.get("query_type", "unknown")

        # 1. CACHE HIT CHECK
        cached_response = self.response_cache.get(messages)
        if cached_response:
            print("[CACHE HIT] Returning cached response")
            return {"messages": [cached_response]}
            
        # 2. FAST PATH CHECK
        is_complex_query = query_type in ['stock', 'general', 'knowledge']

        if not is_complex_query:
            message_type = self.fast_path_router.is_generic_message(message_content)
            
            if message_type:
                print(f"[FAST PATH] Handling {message_type}")
                fast_response_content = self.fast_path_router.get_fast_response(message_type)
                
                # Store in Redis
                self.memory_manager.store_message(session_id, "assistant", fast_response_content)
                
                return {"messages": [AIMessage(content=fast_response_content)]}

        # 3. FULL LLM EXECUTION WITH MEMORY CONTEXT
        
        # Get short-term memory from Redis (last 10 messages)
        redis_history = self.memory_manager.get_short_term_memory(session_id, limit=999)
        
        # Get long-term context from Weaviate
        long_term_context = self.memory_manager.get_long_term_context(user_id, limit=5)
        
        # Get user profile from Weaviate
        user_profile = self.memory_manager.weaviate_manager.get_user_profile(user_id)
        profile_context = ""
        if user_profile:
            profile_context = f"\nUser Profile: {user_profile.get('name', 'Unknown')}"
            if user_profile.get('favorite_stocks'):
                profile_context += f"\nFavorite Stocks: {', '.join(user_profile['favorite_stocks'])}"
            if user_profile.get('trading_goals'):
                profile_context += f"\nTrading Goals: {user_profile['trading_goals']}"
        
        # Build enhanced system message with memory context
        current_summary = state.get("summary", "No prior conversation history.")
        
        enhanced_system_msg = (
            self.system_message.content + 
            f"\n\n--- CONVERSATION CONTEXT ---"
            f"\n{current_summary}"
            f"\n\n--- LONG-TERM MEMORY ---"
            f"\n{long_term_context}"
            f"{profile_context}"
            f"\n\nQUERY TYPE: {query_type.upper()}"
        )
        
        # Use Redis history + current message
        messages_with_system = [SystemMessage(content=enhanced_system_msg)] + redis_history[-10:]
        
        try:
            response = self._invoke_llm_with_retry(messages_with_system)
            self.response_cache.set(messages, response)
            
            # Store assistant response in Redis
            if hasattr(response, 'content'):
                self.memory_manager.store_message(session_id, "assistant", response.content)
            
            return {"messages": [response]}
            
        except Exception as e:
            print(f"[ERROR] LLM invocation failed: {e}")
            error_response = AIMessage(
                content="I'm experiencing high demand right now. Please try again in a moment."
            )
            return {"messages": [error_response]}
    
    def _grade_documents_node(self, state: State):
        """Grade retrieved documents for relevance"""
        messages = state["messages"]
        
        if not messages or len(messages) < 2:
            return state
        
        last_message = messages[-1]
        
        if hasattr(last_message, 'name') and last_message.name == 'retriever_tool':
            content = str(last_message.content)
            query = messages[-2].content if len(messages) >= 2 else ""
            relevance_score = self._calculate_relevance(query, content)
            
            print(f"[CAG] Document relevance score: {relevance_score:.2f}")
            
            needs_correction = relevance_score < 0.5
            
            return {
                **state,
                "relevance_score": relevance_score,
                "needs_correction": needs_correction
            }
        
        return state
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _correction_node(self, state: State):
        """If RAG results are poor, switch to web search"""
        if state.get("needs_correction", False):
            print("[CAG] Low relevance detected, triggering web search correction")
            
            messages = state["messages"]
            query = messages[-2].content if len(messages) >= 2 else ""
            
            correction_msg = HumanMessage(
                content=f"The knowledge base didn't have good information. Search the web for: {query}"
            )
            
            return {"messages": [correction_msg]}
        
        return state
    
    def _should_correct(self, state: State) -> str:
        """Decide if correction is needed"""
        if state.get("needs_correction", False):
            return "correct"
        return "continue"
    
    """def _router_to_tools_or_summarizer(self, state: State) -> str:
        ""Route from chatbot to tools or summarizer""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "summarizer"""
    def _router_to_tools_or_summarizer(self, state: State) -> str:
        """Route from chatbot to tools or summarizer"""
        last_message = state["messages"][-1]
        
        # 1. Check for tool calls (highest priority)
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # 2. Default: Always route to the summarizer node. 
        # The summarizer node will decide whether to archive (LTM) or just END (STM).
        return "summarizer"
    """def _summarization_node(self, state: State):
        ""
        Node to update the running conversation summary and archive to Weaviate
        ""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "anonymous")
        
        current_summary = state.get("summary", "The conversation has just begun.")
        
        # Filter conversational messages
        conversational_messages = [
            m for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and 
               (not hasattr(m, 'tool_calls') or not m.tool_calls)
        ]
        
        new_messages = conversational_messages[-8:]
        
        if not new_messages:
            return state

        formatted_new_messages = "\n".join([
            f"{type(m).__name__}: {m.content}" for m in new_messages
        ])

        try:
            # Update summary
            new_summary_result = self.summary_chain.invoke({
                "current_summary": current_summary,
                "new_messages": formatted_new_messages
            })
            
            new_summary = new_summary_result['text'].strip()
            print(f"[SUMMARY] Summary updated.")
            
            # Extract key topics and user preferences for Weaviate
            key_topics = self._extract_topics(new_messages)
            user_preferences = self._extract_preferences(new_messages)
            
            # Archive to Weaviate every 10 messages or at session end
            if len(conversational_messages) % 10 == 0:
                print("[ARCHIVING] Storing conversation in Weaviate")
                self.memory_manager.archive_conversation(
                    session_id=session_id,
                    summary=new_summary,
                    key_topics=key_topics,
                    user_preferences=user_preferences,
                    user_id=user_id
                )
            
            return {"summary": new_summary}
            
        except Exception as e:
            print(f"[SUMMARY ERROR] Failed to generate summary: {e}")
            return state"""
    def _summarization_node(self, state: State):
        """
        Node to update the running conversation summary and archive to Weaviate
        """
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "anonymous")
        
        current_summary = state.get("summary", "The conversation has just begun.")
        
        # Filter conversational messages
        conversational_messages = [
            m for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and 
               (not hasattr(m, 'tool_calls') or not m.tool_calls)
        ]
        
        # Focus on the last few messages for summarization/extraction
        new_messages = conversational_messages[-8:]
        
        if not new_messages:
            return state

        formatted_new_messages = "\n".join([
            f"{type(m).__name__}: {m.content}" for m in new_messages
        ])

        try:
            # 1. Generate Summary
            new_summary_result = self.summary_chain.invoke({
                "current_summary": current_summary,
                "new_messages": formatted_new_messages
            })
            
            new_summary = new_summary_result['text'].strip()
            print(f"[SUMMARY] Summary updated.")
            
            # 2. Extract key topics and user preferences
            key_topics = self._extract_topics(new_messages)
            user_preferences = self._extract_preferences(new_messages)
            
            # --- START CRITICAL ARCHIVAL LOGIC ---
            
            # Condition A: Long conversation (pruning/archival threshold)
            is_long_conversation = len(conversational_messages) >= 10
            
            # Condition B: Contains specific user profile data (name, risk tolerance, etc.)
            has_significant_preferences = (
                user_preferences.get('name') is not None or 
                len(user_preferences.get('mentioned_stocks', [])) > 0
            )

            # Condition C: Contains relevant topics (covers single stock queries, educational Qs)
            has_relevant_topics = len(key_topics) > 0 and len(conversational_messages) > 1
            
            # Archive if ANY of the conditions are met:
            if is_long_conversation or has_significant_preferences or has_relevant_topics:
                
                # LTM ARCHIVAL (Write to Weaviate)
                print(f"[ARCHIVING] Storing conversation in Weaviate for user {user_id}")
                self.memory_manager.archive_conversation(
                    session_id=session_id,
                    summary=new_summary,
                    key_topics=key_topics,
                    user_preferences=user_preferences,
                    user_id=user_id
                )
                
                # STM PRUNING (Clear Redis)
                print(f"[PRUNING] Clearing short-term memory (Redis) for session {session_id}")
                self.memory_manager.redis_manager.clear_session(session_id)
                
            else:
                # Session is too short/trivial (e.g., "Hi" -> "Hello"). Let Redis TTL handle it.
                print("[SKIPPING ARCHIVAL] Session too trivial. Keeping in Redis until TTL.")

            # --- END CRITICAL ARCHIVAL LOGIC ---

            return {"summary": new_summary}
            
        except Exception as e:
            print(f"[SUMMARY ERROR] Failed to generate summary: {e}")
            return state
    def _extract_topics(self, messages) -> list:
        """Extract key topics from messages"""
        # Simple keyword extraction
        topics = set()
        keywords = ['stock', 'market', 'trading', 'invest', 'portfolio', 'price']
        
        for msg in messages:
            content = msg.content.lower()
            for keyword in keywords:
                if keyword in content:
                    topics.add(keyword)
        
        return list(topics)[:5]
    
    def _extract_preferences(self, messages) -> dict:
        """Extract user preferences from messages"""
        preferences = {}
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                
                # Extract name
                if 'my name is' in content or "i'm" in content or "i am" in content:
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ['name', 'am', "i'm"] and i + 1 < len(words):
                            preferences['name'] = words[i + 1].strip('.,!?')
                
                # Extract stock mentions
                stock_pattern = r'\b[A-Z]{2,5}\b'
                stocks = re.findall(stock_pattern, msg.content)
                if stocks:
                    preferences.setdefault('mentioned_stocks', []).extend(stocks)
        
        return preferences
    
    def build(self):
        """Build the graph with memory integration"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("classifier", self._query_classifier_node)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("grader", self._grade_documents_node)
        graph_builder.add_node("corrector", self._correction_node)
        graph_builder.add_node("summarizer", self._summarization_node)
        
        # Build graph flow
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_edge("classifier", "chatbot")
        graph_builder.add_conditional_edges("chatbot", self._router_to_tools_or_summarizer) 
        graph_builder.add_edge("tools", "grader")
        
        graph_builder.add_conditional_edges(
            "grader",
            self._should_correct,
            {
                "correct": "corrector",
                "continue": "summarizer"
            }
        )
        graph_builder.add_edge("corrector", "chatbot")
        graph_builder.add_edge("summarizer", END)
        
        self.graph = graph_builder.compile()
    
    def get_graph(self):
        if self.graph is None:
            raise ValueError("Graph not built. Call build() first.")
        return self.graph
    
    def clear_cache(self):
        self.response_cache.cache.clear()
    
    def get_cache_stats(self):
        return {
            'response_cache_size': len(self.response_cache.cache)
        }
    
    def close(self):
        """Close all memory connections"""
        self.memory_manager.close()
