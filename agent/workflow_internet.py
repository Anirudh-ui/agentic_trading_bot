"""
Optimized Internet Workflow - Ultra-Fast Routing & Response
- Instant fast-path for greetings (0 LLM calls, <50ms)
- Smart intent detection (News/Stock/General)
- Tavily for news, Yahoo Finance for stocks
- STM-only with session cleanup
- Robust system prompts per query type
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from typing import Dict, List, Optional
import os
import sys
import re
from datetime import datetime

from utils.model_loaders import ModelLoader
from utils.memory_manager import RedisMemoryManager
from utils.cache_manager import get_cache_manager
from toolkit.tools import tavilytool, yahoo_finance_tool
from custom_logging.my_logger import logger, log_execution_time
from exception.exceptions import WorkflowException


class InternetState(TypedDict):
    """State for internet workflow"""
    messages: Annotated[list, add_messages]
    query_type: str  # 'greeting', 'stock', 'news', 'general'
    session_id: str
    is_fast_path: bool
    tickers: List[str]  # Extracted ticker symbols
    needs_tools: bool


class FastPathRouter:
    """
    Lightning-fast pattern matching for common queries
    NO LLM calls - pure regex matching for <50ms responses
    """
    
    # Comprehensive greeting patterns
    GREETING_PATTERNS = [
        r'^(hi|hello|hey|hola|sup|yo)[\s!?.]*$',
        r'^(good\s+(morning|afternoon|evening|day))[\s!?.]*$',
        r'^(what\'?s\s+up|howdy|greetings?)[\s!?.]*$',
    ]
    
    FAREWELL_PATTERNS = [
        r'\b(bye|goodbye|see\s+you|farewell|take\s+care|later|cya)\b',
    ]
    
    THANKS_PATTERNS = [
        r'\b(thanks?|thank\s+you|thx|ty|appreciated|grateful)\b',
    ]
    
    HELP_PATTERNS = [
        r'\b(help|what\s+can\s+you\s+do|capabilities|how\s+do\s+you\s+work)\b',
    ]
    
    # Pre-compiled patterns for speed
    _compiled_patterns = None
    
    @classmethod
    def _compile_patterns(cls):
        """Compile patterns once for faster matching"""
        if cls._compiled_patterns is None:
            cls._compiled_patterns = {
                'greeting': [re.compile(p, re.IGNORECASE) for p in cls.GREETING_PATTERNS],
                'farewell': [re.compile(p, re.IGNORECASE) for p in cls.FAREWELL_PATTERNS],
                'thanks': [re.compile(p, re.IGNORECASE) for p in cls.THANKS_PATTERNS],
                'help': [re.compile(p, re.IGNORECASE) for p in cls.HELP_PATTERNS],
            }
    
    @classmethod
    def classify(cls, text: str) -> Optional[str]:
        """
        Ultra-fast pattern matching
        Returns: 'greeting', 'farewell', 'thanks', 'help', or None
        """
        cls._compile_patterns()
        
        text_stripped = text.strip().lower()
        
        # Check each category
        for category, patterns in cls._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text_stripped):
                    return category
        
        return None
    
    @classmethod
    def get_response(cls, category: str) -> str:
        """Get instant predefined response"""
        responses = {
            'greeting': """👋 **Hello! I'm your Financial Assistant**

I can help you with:

📈 **Stock Information**
• Current prices and performance
• Example: "What's AAPL stock price?"

📰 **Market News**
• Latest financial news and updates
• Example: "Tesla news today"

💹 **Real-time Data**
• Live market information
• Example: "How is MSFT doing?"

What would you like to know?""",

            'farewell': """👋 **Goodbye!**

Thanks for using the Financial Assistant. Come back anytime for:
• Stock prices and analysis
• Market news and trends
• Real-time financial data

Have a great day! 📈""",

            'thanks': """😊 **You're welcome!**

Happy to help! Feel free to ask about:
• Stock prices and performance
• Latest market news
• Company updates and trends

Just ask away!""",

            'help': """💡 **Financial Assistant - Quick Guide**

**📈 Stock Queries:**
• "What's the price of AAPL?"
• "How is Tesla stock doing?"
• "MSFT current price"

**📰 News Queries:**
• "Latest tech stock news"
• "Recent Tesla news"
• "Market trends today"

**💡 Tips:**
• Use ticker symbols (AAPL, TSLA, MSFT) for faster results
• Ask naturally - I understand conversational language
• Be specific about what you want to know

What would you like to explore?"""
        }
        
        return responses.get(category, responses['greeting'])


class SmartIntentClassifier:
    """
    Fast intent classification without LLM
    Determines: Stock vs News vs General
    """
    
    # Stock-related patterns
    STOCK_PATTERNS = [
        r'\b(price|stock|ticker|quote|trading|share)\b',
        r'\b[A-Z]{2,5}\b',  # Ticker symbols
        r'\b(company|corp|inc)\b.*\b(stock|price|performance)\b',
    ]
    
    # News patterns
    NEWS_PATTERNS = [
        r'\b(news|latest|recent|update|announcement|headline)\b',
        r'\b(what\'?s\s+happening|tell\s+me\s+about)\b',
        r'\b(market|industry|sector)\b.*\b(news|update|trend)\b',
    ]
    
    # Company name to ticker mapping (extended)
    COMPANY_TO_TICKER = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'alphabet': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA',
        'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
        'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
        'oracle': 'ORCL', 'ibm': 'IBM', 'intel': 'INTC', 'amd': 'AMD',
    }
    
    @classmethod
    def extract_tickers(cls, text: str) -> List[str]:
        """Extract ticker symbols from text"""
        tickers = set()
        
        # Pattern 1: All-caps ticker symbols (2-5 letters)
        caps_tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        tickers.update(caps_tickers)
        
        # Pattern 2: Company names
        text_lower = text.lower()
        for company, ticker in cls.COMPANY_TO_TICKER.items():
            if company in text_lower:
                tickers.add(ticker)
        
        return list(tickers)
    
    @classmethod
    def classify(cls, text: str) -> Dict[str, any]:
        """
        Fast intent classification
        
        Returns:
            {
                'query_type': 'stock' | 'news' | 'general',
                'tickers': List[str],
                'needs_tools': bool
            }
        """
        text_lower = text.lower()
        tickers = cls.extract_tickers(text)
        
        # Stock query if tickers mentioned + stock keywords
        has_stock_keywords = any(
            re.search(pattern, text_lower, re.IGNORECASE)
            for pattern in cls.STOCK_PATTERNS
        )
        
        # News query if news keywords present
        has_news_keywords = any(
            re.search(pattern, text_lower, re.IGNORECASE)
            for pattern in cls.NEWS_PATTERNS
        )
        
        # Determine query type
        if tickers and has_stock_keywords:
            return {
                'query_type': 'stock',
                'tickers': tickers,
                'needs_tools': True
            }
        elif has_news_keywords:
            return {
                'query_type': 'news',
                'tickers': tickers,
                'needs_tools': True
            }
        else:
            return {
                'query_type': 'general',
                'tickers': tickers,
                'needs_tools': True
            }


class SystemPromptBuilder:
    """Build robust, query-type-specific system prompts"""
    
    BASE_PROMPT = """You are a **Financial Assistant** powered by advanced AI for fast, accurate responses.
    ***Note : do not use specicial characters like <,>,/,* in your response.
    make the response cleaner and adjust the message to fit into the message box***
**CONVERSATION CONTEXT:**
{conversation_history}

**CURRENT DATE:** {current_date}
"""
    
    STOCK_PROMPT = BASE_PROMPT + """
**QUERY TYPE:** Stock Information

**YOUR TASK:**
1. Extract key stock data from tool results
2. Present current price prominently
3. Show change/percentage with visual indicators ( ↑ for up, ↓ for down)
4. Include relevant metrics (volume, market cap if available)
5. Keep response concise and scannable

**FORMAT EXAMPLE:**
Apple Inc (AAPL)
Current Price: $150.25 ↑
Change: +$2.30 (+1.56%)
Volume: 52.3M

TOOL RESULTS:
{tool_results}

USER QUERY: {user_query}
"""
    
    NEWS_PROMPT = BASE_PROMPT + """
**QUERY TYPE:** Market News & Updates

**YOUR TASK:**
1. Summarize key points from news articles
2. Present top 3 most relevant results
3. Include publication dates when available
4. Cite sources: "According to [Source], ..."
5. Use bullet points for clarity

**FORMAT EXAMPLE:**
📰 **Latest Tesla News:**

• **Production Milestone** (Reuters, Dec 2024)
  Tesla reaches 2M vehicle production...

• **Stock Performance** (Bloomberg, Dec 2024)
  Shares up 5% following...

**TOOL RESULTS:**
{tool_results}

**USER QUERY:** {user_query}
"""
    
    GENERAL_PROMPT = BASE_PROMPT + """
**QUERY TYPE:** General Financial Query

**YOUR TASK:**
1. Provide accurate, helpful financial information
2. Use tool results to support your answer
3. Be conversational but professional
4. Cite sources when referencing specific data
5. Keep responses focused and actionable

**TOOL RESULTS:**
{tool_results}

**USER QUERY:** {user_query}
"""
    
    @classmethod
    def build(cls, query_type: str, user_query: str, tool_results: str, history: str) -> str:
        """Build prompt based on query type"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt_templates = {
            'stock': cls.STOCK_PROMPT,
            'news': cls.NEWS_PROMPT,
            'general': cls.GENERAL_PROMPT,
        }
        
        template = prompt_templates.get(query_type, cls.GENERAL_PROMPT)
        
        return template.format(
            conversation_history=history or "No previous conversation",
            current_date=current_date,
            tool_results=tool_results,
            user_query=user_query
        )


class OptimizedInternetWorkflow:
    """
    Optimized Internet Workflow with:
    - Fast-path routing (<50ms for greetings)
    - Smart intent classification
    - Query-specific tool selection
    - STM-only memory management
    - Automatic cleanup on session close
    """
    
    def __init__(self):
        """Initialize workflow components"""
        try:
            logger.info("[WORKFLOW] Initializing optimized internet workflow...")
            
            # LLM (Groq for speed)
            self.model_loader = ModelLoader()
            self.llm = self.model_loader.load_llm()
            logger.info("[WORKFLOW] Groq LLM loaded")
            
            # Redis STM only
            self._setup_redis()
            
            # Tools
            self.tools = [tavilytool, yahoo_finance_tool]
            logger.info("[WORKFLOW] Tools loaded: Tavily, Yahoo Finance")
            
            # Cache manager
            self.cache_manager = get_cache_manager(ttl_seconds=1800)
            
            self.graph = None
            
            logger.info("[WORKFLOW] Initialization complete")
            
        except Exception as e:
            logger.error(f"[WORKFLOW] Initialization failed: {e}")
            raise WorkflowException("Failed to initialize workflow", sys)
    
    def _setup_redis(self):
        """Setup Redis for STM"""
        try:
            self.redis_memory = RedisMemoryManager(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6380)),
                ttl_hours=2  # Auto-expire after 2 hours
            )
            logger.info("[REDIS] Connected for STM")
            
        except Exception as e:
            logger.error(f"[REDIS] Setup failed: {e}")
            raise WorkflowException("Failed to setup Redis", sys)
    
    @log_execution_time
    def _fast_path_node(self, state: InternetState) -> InternetState:
        """
        Node 1: Lightning-fast greeting detection
        Target: <50ms response time
        """
        try:
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if not last_message or not hasattr(last_message, 'content'):
                return state
            
            query = last_message.content.strip()
            
            # Fast-path classification
            fast_category = FastPathRouter.classify(query)
            
            if fast_category:
                logger.info(f"[FAST-PATH] ⚡ Matched: {fast_category}")
                return {
                    **state,
                    "is_fast_path": True,
                    "query_type": fast_category,
                    "needs_tools": False
                }
            
            logger.info("[FAST-PATH] No match - proceeding to intent classification")
            return {
                **state,
                "is_fast_path": False
            }
            
        except Exception as e:
            logger.error(f"[FAST-PATH] Error: {e}")
            return state
    
    @log_execution_time
    def _intent_classification_node(self, state: InternetState) -> InternetState:
        """
        Node 2: Smart intent classification (Stock/News/General)
        Target: <100ms classification time
        """
        try:
            # Skip if fast-path
            if state.get("is_fast_path", False):
                return state
            
            messages = state["messages"]
            last_message = messages[-1] if messages else None
            
            if not last_message:
                return state
            
            query = last_message.content
            logger.info(f"[INTENT] Classifying: {query[:60]}...")
            
            # Fast intent classification
            intent_data = SmartIntentClassifier.classify(query)
            
            logger.info(
                f"[INTENT] Type: {intent_data['query_type']} | "
                f"Tickers: {intent_data['tickers']} | "
                f"Needs tools: {intent_data['needs_tools']}"
            )
            
            return {
                **state,
                "query_type": intent_data['query_type'],
                "tickers": intent_data['tickers'],
                "needs_tools": intent_data['needs_tools']
            }
            
        except Exception as e:
            logger.error(f"[INTENT] Classification error: {e}")
            return {
                **state,
                "query_type": "general",
                "needs_tools": True
            }
    
    @log_execution_time
    def _response_node(self, state: InternetState) -> Dict:
        """
        Node 3: Generate response (Fast-path OR LLM+Tools)
        """
        try:
            messages = state["messages"]
            session_id = state.get("session_id", "default")
            query_type = state.get("query_type", "general")
            is_fast_path = state.get("is_fast_path", False)
            
            last_message = messages[-1] if messages else None
            
            if not last_message:
                return {"messages": [AIMessage(content="No query provided.")]}
            
            query = last_message.content
            
            # FAST PATH: Instant response
            if is_fast_path:
                logger.info(f"[RESPONSE] ⚡ Fast-path: {query_type}")
                response_text = FastPathRouter.get_response(query_type)
                
                # Store in STM
                self.redis_memory.store_message(session_id, {
                    'role': 'assistant',
                    'content': response_text
                })
                
                return {"messages": [AIMessage(content=response_text)]}
            
            # NORMAL PATH: Check cache first
            cache_key = self.cache_manager._generate_key(
                query, query_type, prefix="internet_response"
            )
            cached_response = self.cache_manager.get(cache_key)
            
            if cached_response:
                logger.info("[RESPONSE] 💾 Cache hit")
                return {"messages": [cached_response]}
            
            # Get conversation history (last 4 messages for context)
            redis_history = self.redis_memory.get_langchain_messages(session_id, limit=4)
            
            # Format conversation history
            conversation_history = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content[:150]}..."
                for msg in redis_history[-4:]
            ]) if redis_history else "No previous conversation"
            
            # Build query-specific system prompt
            system_msg = SystemPromptBuilder.build(
                query_type=query_type,
                user_query=query,
                tool_results="[Tool results will be inserted after execution]",
                history=conversation_history
            )
            
            # Prepare messages for LLM
            messages_with_system = [SystemMessage(content=system_msg)] + redis_history[-3:]
            
            # Invoke LLM with tools
            logger.info(f"[RESPONSE] 🤖 Generating response for: {query_type}")
            response = self.llm.bind_tools(tools=self.tools).invoke(messages_with_system)
            
            # Cache response
            self.cache_manager.set(cache_key, response)
            
            # Store in STM
            if hasattr(response, 'content'):
                self.redis_memory.store_message(session_id, {
                    'role': 'assistant',
                    'content': response.content
                })
            
            logger.info(f"[RESPONSE] ✓ Generated for {query_type}")
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"[RESPONSE] Error: {e}")
            error_msg = AIMessage(
                content="I'm experiencing technical difficulties. Please try again in a moment."
            )
            return {"messages": [error_msg]}
    
    def _route_to_tools_or_end(self, state: InternetState) -> str:
        """Conditional edge: Route to tools or end"""
        # Fast-path always skips tools
        if state.get("is_fast_path", False):
            return "end"
        
        # Check if LLM called tools
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"[ROUTER] → tools ({len(last_message.tool_calls)} calls)")
            return "tools"
        
        return "end"
    
    def build(self):
        """Build optimized workflow graph"""
        try:
            logger.info("[WORKFLOW] Building graph...")
            
            graph_builder = StateGraph(InternetState)
            
            # Add nodes
            graph_builder.add_node("fast_path", self._fast_path_node)
            graph_builder.add_node("intent_classification", self._intent_classification_node)
            graph_builder.add_node("response", self._response_node)
            graph_builder.add_node("tools", ToolNode(tools=self.tools))
            
            # Define workflow
            graph_builder.add_edge(START, "fast_path")
            graph_builder.add_edge("fast_path", "intent_classification")
            graph_builder.add_edge("intent_classification", "response")
            
            # Conditional routing
            graph_builder.add_conditional_edges(
                "response",
                self._route_to_tools_or_end,
                {"tools": "tools", "end": END}
            )
            graph_builder.add_edge("tools", "response")
            
            self.graph = graph_builder.compile()
            
            logger.info("[WORKFLOW] ✓ Graph built successfully")
            
        except Exception as e:
            logger.error(f"[WORKFLOW] Graph building failed: {e}")
            raise WorkflowException("Failed to build workflow", sys)
    
    def get_graph(self):
        """Get compiled graph"""
        if not self.graph:
            raise WorkflowException("Graph not built. Call build() first.", sys)
        return self.graph
    
    def clear_session(self, session_id: str):
        """Clear session STM and cache on chat close"""
        logger.info(f"[CLEANUP] 🧹 Clearing session: {session_id}")
        
        # Clear STM
        self.redis_memory.clear_session(session_id)
        
        # Clear related cache entries
        # (Optional: clear cache entries with session_id prefix)
        
        logger.info(f"[CLEANUP] ✓ Session cleared: {session_id}")
    
    def clear_cache(self):
        """Clear all response cache"""
        count = self.cache_manager.clear(prefix="internet_response")
        logger.info(f"[CACHE] 🗑️  Cleared {count} cache entries")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache_manager.get_stats()


# Alias for backward compatibility
InternetWorkflowBuilder = OptimizedInternetWorkflow

# Export
__all__ = ['OptimizedInternetWorkflow', 'InternetWorkflowBuilder']