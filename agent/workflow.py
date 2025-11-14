from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.tools import *
from toolkit.smart_router import SmartToolRouter
import re
from functools import lru_cache
import hashlib
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str  # 'stock', 'general', 'knowledge'
    relevance_score: float
    needs_correction: bool

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
        
        # Tools with clear purposes
        self.tools = [retriever_tool, yahoo_finance_tool, tavilytool]
        
        # System prompt for better tool calling
        system_prompt = """You are a financial trading assistant. Follow these rules STRICTLY:

1. STOCK QUERIES (prices, financials, company data):
   - ALWAYS use yahoo_finance_tool
   - Examples: "AAPL stock price", "Tesla earnings", "Microsoft market cap"

2. GENERAL/RECENT INFO (news, current events):
   - ALWAYS use tavilytool (Tavily search)
   - Examples: "latest crypto news", "market trends today", "Fed interest rate decision"

3. KNOWLEDGE BASE (trading strategies, concepts, historical info):
   - ONLY use retriever_tool as LAST RESORT
   - Examples: "what is a bull market", "explain RSI indicator"

4. NEVER call multiple tools for the same information
5. NEVER call retriever_tool if yahoo_finance_tool or tavilytool can answer
6. Be concise and accurate

Choose the RIGHT tool based on the query type."""

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
        
        # Use smart router to classify
        query_type = self.smart_router.classify_query(query)
        
        print(f"[ROUTER] Query classified as: {query_type}")
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _chatbot_node(self, state: State):
        """Enhanced chatbot node with caching and smart routing"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if last_message and hasattr(last_message, 'content'):
            message_content = last_message.content
            
            # Fast path: Check for generic messages
            message_type = self.fast_path_router.is_generic_message(message_content)
            if message_type:
                fast_response = self.fast_path_router.get_fast_response(message_type)
                return {"messages": [AIMessage(content=fast_response)]}
            
            # Check cache
            cached_response = self.response_cache.get(messages)
            if cached_response:
                print("[CACHE HIT] Returning cached response")
                return {"messages": [cached_response]}
        
        # Add system message with query type hint
        query_type = state.get("query_type", "unknown")
        enhanced_system_msg = self.system_message.content + f"\n\nQUERY TYPE: {query_type.upper()}"
        
        messages_with_system = [SystemMessage(content=enhanced_system_msg)] + messages
        
        try:
            response = self._invoke_llm_with_retry(messages_with_system)
            self.response_cache.set(messages, response)
            return {"messages": [response]}
        except Exception as e:
            error_response = AIMessage(
                content=f"I'm experiencing high demand right now. Please try again in a moment."
            )
            return {"messages": [error_response]}
    
    def _grade_documents_node(self, state: State):
        """Grade retrieved documents for relevance (CAG - Corrective step)"""
        messages = state["messages"]
        
        # Check if last message contains tool results
        if not messages or len(messages) < 2:
            return state
        
        last_message = messages[-1]
        
        # If it's a ToolMessage from retriever, grade it
        if hasattr(last_message, 'name') and last_message.name == 'retriever_tool':
            content = str(last_message.content)
            
            # Simple relevance check
            query = messages[-2].content if len(messages) >= 2 else ""
            relevance_score = self._calculate_relevance(query, content)
            
            print(f"[CAG] Document relevance score: {relevance_score:.2f}")
            
            needs_correction = relevance_score < 0.5  # Threshold
            
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
        
        # Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _correction_node(self, state: State):
        """If RAG results are poor, switch to web search (Corrective Action)"""
        if state.get("needs_correction", False):
            print("[CAG] Low relevance detected, triggering web search correction")
            
            messages = state["messages"]
            query = messages[-2].content if len(messages) >= 2 else ""
            
            # Force web search
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
    
    def build(self):
        """Build the CAG graph with smart routing"""
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("classifier", self._query_classifier_node)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("grader", self._grade_documents_node)
        graph_builder.add_node("corrector", self._correction_node)
        
        # Build graph flow
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_edge("classifier", "chatbot")
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "grader")
        
        # CAG: Conditional edge based on relevance
        graph_builder.add_conditional_edges(
            "grader",
            self._should_correct,
            {
                "correct": "corrector",
                "continue": "chatbot"
            }
        )
        graph_builder.add_edge("corrector", "chatbot")
        
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