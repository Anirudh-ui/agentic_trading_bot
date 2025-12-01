from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from utils.model_loaders import ModelLoader
from toolkit.enhanced_comparison_tools import *
from toolkit.enhanced_smart_router import EnhancedSmartRouter
from toolkit.tools import extract_financial_entities, tavilytool
from utils.memory_manager import HybridMemoryManager
import re
from functools import lru_cache
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_classic.chains import LLMChain
from langchain_classic.prompts import PromptTemplate
import os
import hashlib
import json


class State(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: str
    relevance_score: float
    needs_correction: bool
    summary: str
    session_id: str
    user_id: str
    extracted_tickers: list  # For comparison routing
    is_comparison: bool  # Flag for comparison queries
    comparison_results: dict  # Store intermediate comparison results


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
        'greeting': "Hello! I'm your trading assistant with access to:\n- Real-time stock prices and comparisons\n- Multi-modal knowledge base (text, images, tables)\n- Latest financial news\n- Trading strategies\n\nWhat would you like to explore?",
        'farewell': "Goodbye! Feel free to return for stock analysis or trading insights.",
        'thanks': "You're welcome! Let me know if you need more help with stocks or trading.",
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


class EnhancedGraphBuilder:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        
        # Initialize enhanced router
        self.smart_router = EnhancedSmartRouter()
        
        # Initialize hybrid memory manager
        self.memory_manager = HybridMemoryManager(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', 6380)),
            weaviate_url=os.getenv('WEAVIATE_URL', 'http://localhost:8080'),
            weaviate_api_key=os.getenv('WEAVIATE_API_KEY'),
            session_ttl_hours=24
        )
        
        # Summary prompt
        summary_prompt = PromptTemplate.from_template(
            """
            Progressively summarize the conversation for a financial trading assistant. 
            Keep under 100 words and capture:
            - User's preferences and goals
            - Stocks discussed and compared
            - Key insights and decisions
            
            Current Summary: {current_summary}
            New Messages: {new_messages}
            
            Return ONLY the updated summary.
            """
        )
        self.summary_chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        
        # Enhanced tool set
        self.tools = [
            multimodal_retriever_tool,
            compare_stocks_tool,
            get_stock_price_tool,
            tavilytool,
            extract_financial_entities
        ]
        
        # Enhanced system prompt
        system_prompt = """You are an **Expert Stock Market Analyst** with multi-modal analysis capabilities.

**CRITICAL AGENT PROTOCOLS (EXECUTE IN ORDER):**

**1. QUERY ANALYSIS & ENTITY EXTRACTION:**
- **MANDATE:** For ANY query with stock names, tickers, or financial terms, FIRST call `extract_financial_entities` to extract structured data.
- **PURPOSE:** Convert natural language to structured tags (COMPANY, TICKER, DATE, AMOUNT) for precise tool calls.
- **DO NOT SKIP THIS STEP** for stock-related queries.

**2. COMPARISON QUERIES (HIGHEST PRIORITY):**
- **DETECTION:** User asks to "compare", mentions "vs", "versus", "difference between", OR mentions 2+ stocks.
- **PROCEDURE:**
  a. Extract entities first using `extract_financial_entities`
  b. Identify ALL tickers mentioned (minimum 2 required)
  c. Call `compare_stocks_tool` with JSON: {{"tickers": ["TICKER1", "TICKER2", ...], "period": "1mo"}}
  d. Present comprehensive comparison report
- **EXAMPLES:**
  * "Compare Apple and Tesla" → Extract entities → compare_stocks_tool({{"tickers": ["AAPL", "TSLA"], "period": "1mo"}})
  * "How do MSFT, GOOGL, and AMZN compare?" → compare_stocks_tool({{"tickers": ["MSFT", "GOOGL", "AMZN"], "period": "1mo"}})

**3. SINGLE STOCK QUERIES:**
- **DETECTION:** User asks about ONE specific stock's price/info
- **PROCEDURE:**
  a. Extract ticker using `extract_financial_entities`
  b. Call `get_stock_price_tool` with the ticker
- **EXAMPLE:** "What's Apple's stock price?" → get_stock_price_tool("AAPL")

**4. KNOWLEDGE BASE QUERIES:**
- **DETECTION:** Questions about strategies, concepts, charts, tables, or documents
- **PROCEDURE:**
  a. Call `multimodal_retriever_tool` to search across text, images, and tables
  b. If no results, fall back to web search
- **EXAMPLES:**
  * "What's a bull market strategy?" → multimodal_retriever_tool
  * "Show me the chart about dividends" → multimodal_retriever_tool
  * "Explain the table on page 5" → multimodal_retriever_tool

**5. NEWS & CURRENT EVENTS:**
- **DETECTION:** "latest", "news", "recent developments", "what's happening"
- **PROCEDURE:** Call `tavilytool` for current information

**6. FALLBACK CHAIN:**
If primary tool fails:
1. Knowledge base → Web search → Direct answer
2. Always try knowledge base for trading concepts BEFORE web search

**MEMORY CONTEXT:**
- **User Profile:** {profile_context}
- **Long-Term Context:** {long_term_context}
- **Recent History:** {redis_history}

**RESPONSE GUIDELINES:**
- Be concise but comprehensive for comparisons
- Use data to support all claims
- Format numbers clearly (prices, percentages, market caps)
- Highlight key insights for investment decisions
- If tools fail, acknowledge and provide best alternative

**CURRENT QUERY TYPE:** {query_type}
**TICKERS DETECTED:** {tickers}
"""

        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.system_message = SystemMessage(content=system_prompt)
        self.graph = None
        
        # Caching
        self.response_cache = ResponseCache(ttl_seconds=1800)
        self.fast_path_router = FastPathRouter()
    
    def _query_classifier_node(self, state: State):
        """Enhanced classifier with comparison detection"""
        messages = state["messages"]
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, 'content'):
            return state
        
        query = last_message.content
        
        # Get comprehensive intent analysis
        intent = self.smart_router.extract_query_intent(query)
        
        query_type = intent['query_type']
        tickers = intent['tickers']
        is_comparison = intent['is_comparison']
        
        print(f"[ROUTER] Query Type: {query_type}")
        print(f"[ROUTER] Tickers: {tickers}")
        print(f"[ROUTER] Is Comparison: {is_comparison}")
        
        return {
            **state,
            "query_type": query_type,
            "extracted_tickers": tickers,
            "is_comparison": is_comparison
        }
    
    def _comparison_handler_node(self, state: State):
        """
        Dedicated node for handling comparison queries
        Ensures proper sequential tool calls for each stock
        """
        messages = state["messages"]
        tickers = state.get("extracted_tickers", [])
        
        if len(tickers) < 2:
            print("[COMPARISON] Insufficient tickers for comparison")
            return state
        
        print(f"[COMPARISON] Handling comparison for: {', '.join(tickers)}")
        
        # Build comparison tool call
        comparison_request = {
            "tickers": tickers,
            "period": state.get("time_period", "1mo")
        }
        
        # Create a message that will trigger the comparison tool
        comparison_query = HumanMessage(
            content=f"Compare these stocks comprehensively: {', '.join(tickers)}"
        )
        
        return {"messages": [comparison_query]}
    
    def _chatbot_node(self, state: State):
        """Enhanced chatbot with comparison routing"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "anonymous")
        query_type = state.get("query_type", "unknown")
        tickers = state.get("extracted_tickers", [])
        is_comparison = state.get("is_comparison", False)
        
        last_message = messages[-1] if messages else None
        
        if not last_message or not hasattr(last_message, 'content'):
            return state

        message_content = last_message.content

        # 1. CACHE CHECK
        cached_response = self.response_cache.get(messages)
        if cached_response:
            print("[CACHE HIT]")
            return {"messages": [cached_response]}
            
        # 2. FAST PATH CHECK
        is_complex = query_type in ['comparison', 'stock_price', 'knowledge_base', 'multimodal', 'news']
        
        if not is_complex:
            message_type = self.fast_path_router.is_generic_message(message_content)
            if message_type:
                print(f"[FAST PATH] {message_type}")
                fast_response = self.fast_path_router.get_fast_response(message_type)
                self.memory_manager.store_message(session_id, "assistant", fast_response)
                return {"messages": [AIMessage(content=fast_response)]}

        # 3. FULL LLM WITH ENHANCED CONTEXT
        redis_history = self.memory_manager.get_short_term_memory(session_id, limit=999)
        long_term_context = self.memory_manager.get_long_term_context(user_id, limit=5)
        
        user_profile = self.memory_manager.weaviate_manager.get_user_profile(user_id)
        profile_context = ""
        if user_profile:
            profile_context = f"User: {user_profile.get('name', 'Unknown')}"
            if user_profile.get('favorite_stocks'):
                profile_context += f" | Favorites: {', '.join(user_profile['favorite_stocks'])}"
        
        current_summary = state.get("summary", "New conversation")
        
        # Build enhanced system message
        enhanced_system_msg = self.system_message.content.format(
            profile_context=profile_context or "New user",
            long_term_context=long_term_context,
            redis_history="Recent conversation available",
            query_type=query_type.upper(),
            tickers=", ".join(tickers) if tickers else "None detected"
        )
        
        messages_with_system = [SystemMessage(content=enhanced_system_msg)] + redis_history[-10:]
        
        try:
            response = self.llm_with_tools.invoke(messages_with_system)
            self.response_cache.set(messages, response)
            
            if hasattr(response, 'content'):
                self.memory_manager.store_message(session_id, "assistant", response.content)
            
            return {"messages": [response]}
            
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            return {"messages": [AIMessage(content="I'm experiencing high demand. Please retry.")]}
    
    def _grade_documents_node(self, state: State):
        """Grade retrieved documents"""
        messages = state["messages"]
        
        if not messages or len(messages) < 2:
            return state
        
        last_message = messages[-1]
        
        if hasattr(last_message, 'name') and 'retriever' in str(last_message.name).lower():
            content = str(last_message.content)
            query = messages[-2].content if len(messages) >= 2 else ""
            relevance_score = self._calculate_relevance(query, content)
            
            print(f"[GRADER] Relevance: {relevance_score:.2f}")
            
            needs_correction = relevance_score < 0.5
            
            return {
                **state,
                "relevance_score": relevance_score,
                "needs_correction": needs_correction
            }
        
        return state
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _correction_node(self, state: State):
        """Fallback to web search if RAG fails"""
        if state.get("needs_correction", False):
            print("[CORRECTOR] Triggering web search fallback")
            
            messages = state["messages"]
            query = messages[-2].content if len(messages) >= 2 else ""
            
            correction_msg = HumanMessage(
                content=f"Knowledge base had insufficient info. Search web for: {query}"
            )
            
            return {"messages": [correction_msg]}
        
        return state
    
    def _should_correct(self, state: State) -> str:
        """Route based on relevance"""
        return "correct" if state.get("needs_correction", False) else "continue"
    
    def _router_to_tools_or_summarizer(self, state: State) -> str:
        """Route from chatbot"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        return "summarizer"
    
    def _summarization_node(self, state: State):
        """Summarization with intelligent archiving"""
        messages = state["messages"]
        session_id = state.get("session_id", "default")
        user_id = state.get("user_id", "anonymous")
        
        current_summary = state.get("summary", "New conversation")
        
        conversational_messages = [
            m for m in messages 
            if isinstance(m, (HumanMessage, AIMessage)) and 
               (not hasattr(m, 'tool_calls') or not m.tool_calls)
        ]
        
        new_messages = conversational_messages[-8:]
        
        if not new_messages:
            return state

        formatted = "\n".join([f"{type(m).__name__}: {m.content}" for m in new_messages])

        try:
            new_summary_result = self.summary_chain.invoke({
                "current_summary": current_summary,
                "new_messages": formatted
            })
            
            new_summary = new_summary_result['text'].strip()
            print(f"[SUMMARY] Updated")
            
            key_topics = self._extract_topics(new_messages)
            user_preferences = self._extract_preferences(new_messages)
            
            # Archive if significant
            is_long = len(conversational_messages) >= 10
            has_preferences = bool(user_preferences.get('name') or user_preferences.get('mentioned_stocks'))
            has_topics = len(key_topics) > 0 and len(conversational_messages) > 1
            
            if is_long or has_preferences or has_topics:
                print(f"[ARCHIVING] Session {session_id}")
                self.memory_manager.archive_conversation(
                    session_id=session_id,
                    summary=new_summary,
                    key_topics=key_topics,
                    user_preferences=user_preferences,
                    user_id=user_id
                )
            else:
                print("[SKIP ARCHIVING] Session too trivial")

            return {"summary": new_summary}
            
        except Exception as e:
            print(f"[SUMMARY ERROR] {e}")
            return state
    
    def _extract_topics(self, messages) -> list:
        """Extract topics"""
        topics = set()
        keywords = ['stock', 'market', 'trading', 'comparison', 'price', 'invest']
        
        for msg in messages:
            content = msg.content.lower()
            for keyword in keywords:
                if keyword in content:
                    topics.add(keyword)
        
        return list(topics)[:5]
    
    def _extract_preferences(self, messages) -> dict:
        """Extract preferences"""
        preferences = {}
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                
                if 'my name is' in content or "i'm" in content:
                    words = content.split()
                    for i, word in enumerate(words):
                        if word in ['name', 'am', "i'm"] and i + 1 < len(words):
                            preferences['name'] = words[i + 1].strip('.,!?')
                
                stock_pattern = r'\b[A-Z]{2,5}\b'
                stocks = re.findall(stock_pattern, msg.content)
                if stocks:
                    preferences.setdefault('mentioned_stocks', []).extend(stocks)
        
        return preferences
    
    def build(self):
        """Build enhanced graph"""
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("classifier", self._query_classifier_node)
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("grader", self._grade_documents_node)
        graph_builder.add_node("corrector", self._correction_node)
        graph_builder.add_node("summarizer", self._summarization_node)
        
        graph_builder.add_edge(START, "classifier")
        graph_builder.add_edge("classifier", "chatbot")
        graph_builder.add_conditional_edges("chatbot", self._router_to_tools_or_summarizer)
        graph_builder.add_edge("tools", "grader")
        graph_builder.add_conditional_edges(
            "grader",
            self._should_correct,
            {"correct": "corrector", "continue": "summarizer"}
        )
        graph_builder.add_edge("corrector", "chatbot")
        graph_builder.add_edge("summarizer", END)
        
        self.graph = graph_builder.compile()
    
    def get_graph(self):
        if not self.graph:
            raise ValueError("Graph not built. Call build() first.")
        return self.graph
    
    def clear_cache(self):
        self.response_cache.cache.clear()
    
    def get_cache_stats(self):
        return {'response_cache_size': len(self.response_cache.cache)}
    
    def close(self):
        self.memory_manager.close()


# Alias for backward compatibility
GraphBuilder = EnhancedGraphBuilder
