import re
from typing import List, Dict, Any
import spacy
from collections import defaultdict


class EnhancedSmartRouter:
    """
    Enhanced router with advanced query classification:
    - Stock comparison detection
    - Multi-modal content detection
    - Entity extraction for comparisons
    - Intent classification
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Spacy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Patterns for different query types
        self.comparison_patterns = [
            r'\b(compare|comparison|versus|vs\.?|difference between)\b',
            r'\b(better|worse|which is)\b.*\b(or|and)\b',
            r'\b(both|all|these)\b.*\b(stocks|companies)\b',
            r'\bhow do\b.*\b(compare|differ|stack up)\b'
        ]
        
        self.knowledge_base_patterns = [
            r'\b(strategy|concept|meaning|definition|explain|what is|how to)\b',
            r'\b(document|pdf|uploaded|knowledge base|from the documents)\b',
            r'\b(chart|graph|table|image|visual|figure)\b',
            r'\b(historical|past|previous|archive)\b'
        ]
        
        self.stock_price_patterns = [
            r'\b(price|quote|current|latest|today)\b',
            r'\b(trading at|stock value)\b',
            r'\b(how much|what\'?s the price)\b'
        ]
        
        self.news_patterns = [
            r'\b(news|latest|recent|happening|update)\b',
            r'\b(market|economic|financial)\b.*\b(event|development)\b'
        ]
    
    def extract_stock_tickers(self, query: str) -> List[str]:
        """
        Extract stock ticker symbols from query
        """
        tickers = []
        
        # Pattern 1: All-caps 2-5 letter words (likely tickers)
        caps_tickers = re.findall(r'\b[A-Z]{2,5}\b', query)
        tickers.extend(caps_tickers)
        
        # Pattern 2: Common company names to ticker mapping
        company_to_ticker = {
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'meta': 'META',
            'facebook': 'META',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'adobe': 'ADBE',
            'salesforce': 'CRM',
            'oracle': 'ORCL',
            'ibm': 'IBM',
            'intel': 'INTC',
            'amd': 'AMD',
            # Indian stocks
            'reliance': 'RELIANCE.NS',
            'tcs': 'TCS.NS',
            'infosys': 'INFY.NS',
            'hdfc': 'HDFCBANK.NS',
            'icici': 'ICICIBANK.NS',
            'wipro': 'WIPRO.NS',
            'bharti': 'BHARTIARTL.NS',
            'airtel': 'BHARTIARTL.NS',
            'itc': 'ITC.NS',
            'sbi': 'SBIN.NS',
        }
        
        query_lower = query.lower()
        for company, ticker in company_to_ticker.items():
            if company in query_lower:
                tickers.append(ticker)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for ticker in tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append(ticker)
        
        return unique_tickers
    
    def is_comparison_query(self, query: str) -> bool:
        """Check if query is asking for comparison"""
        query_lower = query.lower()
        
        # Check explicit comparison patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True
        
        # Check if multiple tickers are mentioned
        tickers = self.extract_stock_tickers(query)
        if len(tickers) >= 2:
            return True
        
        # Check for conjunctions with stock-related words
        if re.search(r'\b(and|or)\b', query_lower):
            stock_words = ['stock', 'company', 'share', 'equity']
            if any(word in query_lower for word in stock_words):
                return True
        
        return False
    
    def classify_query(self, query: str) -> str:
        """
        Classify query into categories:
        - comparison: Compare multiple stocks
        - stock_price: Single stock price query
        - knowledge_base: Query about concepts/strategies from documents
        - multimodal: Query about images/tables/charts
        - news: Current news and events
        - general: General trading questions
        """
        query_lower = query.lower()
        
        # 1. Check for comparison queries (highest priority)
        if self.is_comparison_query(query):
            tickers = self.extract_stock_tickers(query)
            if len(tickers) >= 2:
                return "comparison"
        
        # 2. Check for multi-modal content queries
        multimodal_keywords = ['chart', 'graph', 'table', 'image', 'visual', 'figure', 'diagram']
        if any(keyword in query_lower for keyword in multimodal_keywords):
            return "multimodal"
        
        # 3. Check for knowledge base queries
        for pattern in self.knowledge_base_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "knowledge_base"
        
        # 4. Check for single stock price queries
        tickers = self.extract_stock_tickers(query)
        if len(tickers) == 1:
            for pattern in self.stock_price_patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return "stock_price"
        
        # 5. Check for news queries
        for pattern in self.news_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return "news"
        
        # 6. Default to general if tickers are mentioned
        if tickers:
            return "stock"
        
        return "general"
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract comprehensive intent information from query
        
        Returns:
            Dictionary with:
            - query_type: Classification category
            - tickers: List of extracted tickers
            - is_comparison: Boolean flag
            - entities: Named entities (if spacy available)
            - time_period: Extracted time period (if any)
        """
        query_type = self.classify_query(query)
        tickers = self.extract_stock_tickers(query)
        is_comparison = self.is_comparison_query(query)
        
        # Extract time period
        time_period = "1mo"  # default
        time_patterns = {
            r'\b(today|daily|1d)\b': '1d',
            r'\b(week|weekly|7d|5d)\b': '5d',
            r'\b(month|monthly|1mo|30d)\b': '1mo',
            r'\b(quarter|3 months?|3mo)\b': '3mo',
            r'\b(6 months?|6mo|half year)\b': '6mo',
            r'\b(year|yearly|annual|1y|12mo)\b': '1y'
        }
        
        for pattern, period in time_patterns.items():
            if re.search(pattern, query.lower()):
                time_period = period
                break
        
        # Extract entities using spacy
        entities = []
        if self.nlp:
            doc = self.nlp(query)
            entities = [
                {'text': ent.text, 'label': ent.label_}
                for ent in doc.ents
            ]
        
        return {
            'query_type': query_type,
            'tickers': tickers,
            'is_comparison': is_comparison,
            'time_period': time_period,
            'entities': entities
        }
    
    def get_recommended_tools(self, query: str) -> List[str]:
        """
        Recommend specific tools based on query analysis
        
        Returns list of tool names in priority order
        """
        intent = self.extract_query_intent(query)
        query_type = intent['query_type']
        tickers = intent['tickers']
        
        tools = []
        
        if query_type == "comparison":
            tools.append("compare_stocks_tool")
        
        elif query_type == "stock_price" and len(tickers) == 1:
            tools.append("get_stock_price_tool")
        
        elif query_type in ["knowledge_base", "multimodal"]:
            tools.append("multimodal_retriever_tool")
        
        elif query_type == "news":
            tools.append("tavilytool")
        
        elif query_type == "stock" and tickers:
            # Ambiguous - might need both price and info
            tools.append("get_stock_price_tool")
            tools.append("yahoo_finance_tool")
        
        else:
            # General query - check knowledge base first, then web search
            tools.append("multimodal_retriever_tool")
            tools.append("tavilytool")
        
        return tools
    
    def should_extract_entities(self, query: str) -> bool:
        """
        Determine if financial NER should be called
        """
        query_type = self.classify_query(query)
        
        # Always extract for stock-related queries
        if query_type in ["comparison", "stock_price", "stock"]:
            return True
        
        # Extract if query mentions financial terms
        financial_terms = [
            'stock', 'price', 'market', 'trading', 'invest',
            'company', 'ticker', 'share', 'equity', 'dividend'
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in financial_terms)


# For backward compatibility
class SmartToolRouter(EnhancedSmartRouter):
    """Alias for backward compatibility"""
    pass
