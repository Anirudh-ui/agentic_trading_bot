"""
Smart Query Router - Classifies queries to reduce tool calling confusion
Routes queries to the most appropriate tool based on intent
"""

import re
from typing import Literal

QueryType = Literal['stock', 'general', 'knowledge']

class SmartToolRouter:
    """
    Intelligent router that classifies queries to prevent tool confusion
    """
    
    # Stock-related patterns
    STOCK_PATTERNS = [
        # Price queries
        r'\b(price|cost|value|worth|trading at|stock price)\b.*\b(stock|share|equity)\b',
        r'\b(stock|share|ticker)\b.*\b(price|cost|value)\b',
        
        # Company/ticker symbols
        r'\b[A-Z]{1,5}\b.*\b(stock|share|price|earnings|revenue|market cap)\b',
        r'\b(apple|tesla|microsoft|google|amazon|meta|nvidia|amd|intel)\b.*\b(stock|price|earnings)\b',
        
        # Financial metrics
        r'\b(earnings|revenue|profit|loss|market cap|pe ratio|dividend|eps)\b',
        r'\b(financial|quarterly|annual)\b.*\b(report|results|earnings)\b',
        
        # Market data
        r'\b(open|close|high|low|volume)\b.*\b(stock|market|today)\b',
        r'\b(52 week|year high|year low)\b',
        
        # Company specifics
        r'\b(company|corporation|inc|corp)\b.*\b(worth|value|stock|shares)\b'
    ]
    
    # General/Recent info patterns (news, current events)
    GENERAL_PATTERNS = [
        # News and current events
        r'\b(news|latest|recent|today|current|breaking|update)\b',
        r'\b(what.*happening|what.*going on|tell me about)\b.*\b(today|now|recently)\b',
        
        # Market trends and analysis
        r'\b(market.*trend|trend.*market|market.*outlook|forecast)\b',
        r'\b(bull market|bear market|market.*up|market.*down)\b',
        
        # Federal Reserve and policy
        r'\b(fed|federal reserve|interest rate|monetary policy|inflation)\b',
        
        # Crypto news
        r'\b(crypto|bitcoin|ethereum|cryptocurrency)\b.*\b(news|price|trend|update)\b',
        
        # Economic indicators
        r'\b(gdp|unemployment|cpi|inflation rate|economic)\b.*\b(data|report|news)\b',
        
        # General search indicators
        r'\b(search|find|look up|google|browse)\b',
        r'\b(what is happening|current situation|right now)\b'
    ]
    
    # Knowledge base patterns (concepts, strategies, definitions)
    KNOWLEDGE_PATTERNS = [
        # Trading strategies
        r'\b(strategy|strategies|method|technique|approach)\b.*\b(trading|investing)\b',
        r'\b(how to|how do i|how can i)\b.*\b(trade|invest|analyze|profit)\b',
        
        # Technical indicators
        r'\b(rsi|macd|moving average|bollinger|stochastic|fibonacci|candlestick)\b',
        r'\b(indicator|signal|pattern|chart|technical analysis)\b',
        
        # Concepts and definitions
        r'\b(what is|define|explain|meaning of|concept of)\b',
        r'\b(bull|bear|dividend|option|future|derivative|hedge)\b.*\b(mean|means|definition)\b',
        
        # Investment basics
        r'\b(portfolio|diversification|risk management|asset allocation)\b',
        r'\b(fundamental analysis|value investing|growth stock)\b',
        
        # Market mechanics
        r'\b(how.*market.*work|how.*stock.*work|how.*trading.*work)\b',
        r'\b(bid|ask|spread|liquidity|volatility)\b.*\b(mean|definition|explain)\b'
    ]
    
    def __init__(self):
        self.classification_cache = {}
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify query into stock, general, or knowledge category
        Priority: stock > general > knowledge
        """
        if not query:
            return 'general'
        
        # Check cache first
        query_lower = query.lower().strip()
        if query_lower in self.classification_cache:
            return self.classification_cache[query_lower]
        
        # Check for stock patterns (highest priority)
        for pattern in self.STOCK_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.classification_cache[query_lower] = 'stock'
                return 'stock'
        
        # Check for general/news patterns (medium priority)
        for pattern in self.GENERAL_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.classification_cache[query_lower] = 'general'
                return 'general'
        
        # Check for knowledge patterns (lowest priority)
        for pattern in self.KNOWLEDGE_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                self.classification_cache[query_lower] = 'knowledge'
                return 'knowledge'
        
        # Default to general for ambiguous queries
        self.classification_cache[query_lower] = 'general'
        return 'general'
    
    def get_recommended_tool(self, query: str) -> str:
        """
        Get recommended tool name based on query classification
        """
        query_type = self.classify_query(query)
        
        tool_mapping = {
            'stock': 'yahoo_finance_tool',
            'general': 'tavilytool',
            'knowledge': 'retriever_tool'
        }
        
        return tool_mapping[query_type]
    
    def explain_classification(self, query: str) -> dict:
        """
        Get detailed explanation of why query was classified a certain way
        Useful for debugging
        """
        query_type = self.classify_query(query)
        query_lower = query.lower()
        
        matched_patterns = []
        
        if query_type == 'stock':
            pattern_list = self.STOCK_PATTERNS
        elif query_type == 'general':
            pattern_list = self.GENERAL_PATTERNS
        else:
            pattern_list = self.KNOWLEDGE_PATTERNS
        
        for pattern in pattern_list:
            if re.search(pattern, query_lower, re.IGNORECASE):
                matched_patterns.append(pattern)
        
        return {
            'query': query,
            'classification': query_type,
            'recommended_tool': self.get_recommended_tool(query),
            'matched_patterns': matched_patterns[:3],  # Top 3 matches
            'confidence': 'high' if matched_patterns else 'low'
        }


# Test the router
if __name__ == "__main__":
    router = SmartToolRouter()
    
    test_queries = [
        "What is the current price of AAPL stock?",
        "Latest news about cryptocurrency",
        "Explain what is RSI indicator",
        "Tesla earnings report",
        "What's happening in the market today?",
        "How to use moving averages for trading",
        "MSFT market cap",
        "Fed interest rate decision news",
        "What is a bull market?"
    ]
    
    print("=" * 70)
    print("SMART ROUTER CLASSIFICATION TEST")
    print("=" * 70)
    
    for query in test_queries:
        result = router.explain_classification(query)
        print(f"\nQuery: {query}")
        print(f"Classification: {result['classification']}")
        print(f"Recommended Tool: {result['recommended_tool']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 70)