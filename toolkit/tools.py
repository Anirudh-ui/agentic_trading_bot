import os
from langchain.tools import tool
#from langchain_tavily import TavilySearchResults
from langchain_tavily import TavilySearch 
from data_models.models import RagToolSchema
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from dotenv import load_dotenv
from pinecone import Pinecone
import yfinance as yf
from typing import Optional
import json

load_dotenv()
model_loader = ModelLoader()
config = load_config()

@tool(args_schema=RagToolSchema)
def retriever_tool(question: str):
    
    """
    Knowledge base retrieval tool for trading strategies, concepts, and educational content.
    USE ONLY when yahoo_finance_tool and tavilytool cannot answer the query.
    Best for: trading strategies, technical indicators, market concepts, definitions.
    """
    try:
        print(f"Retriever Tool Invoked with question: {question}")
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        print(pinecone_api_key)
        pc = Pinecone(api_key=pinecone_api_key)
        vector_store = PineconeVectorStore(
            index=pc.Index(config["vector_db"]["index_name"]), 
            embedding=model_loader.load_embeddings()
        )
        
        # Use MMR for diverse results
        retriever = vector_store.as_retriever(
            search_type="mmr",  # Maximum Marginal Relevance for diversity
            search_kwargs={
                "k": config["retriever"]["top_k"],
                "fetch_k": 50,  # Fetch more candidates
                "lambda_mult": 1.0  # Balance between relevance and diversity
            }
        )
        
        retriever_result = retriever.invoke(question)
        print(retriever_result)
        
        if not retriever_result:
            return "No relevant information found in knowledge base. Try using web search for recent information."
        
        # Format results with relevance scores
        formatted_results = []
        for i, doc in enumerate(retriever_result, 1):
            formatted_results.append(f"[Result {i}]\n{doc.page_content}\n")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error retrieving from knowledge base: {str(e)}"


@tool
def yahoo_finance_tool(query: str):
    """
    Real-time stock market data tool using Yahoo Finance.
    USE THIS for: stock prices, company financials, market data, earnings, dividends.
    
    Supports:
    - Stock prices: "AAPL", "TSLA", "MSFT"
    - Company info: market cap, PE ratio, dividends
    - Historical data: 52-week high/low
    - Financial metrics: earnings, revenue, profit
    
    Args:
        query: Stock ticker symbol or company name with specific question
        
    Returns:
        Comprehensive stock data and metrics
    """
    try:
        # Extract ticker symbol from query
        ticker = extract_ticker(query)
        
        if not ticker:
            return "Please provide a valid stock ticker symbol (e.g., AAPL, TSLA, MSFT)"
        
        # Get stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price data
        current_data = stock.history(period="1d")
        
        if current_data.empty:
            return f"No data found for ticker: {ticker}. Please verify the symbol."
        
        # Build comprehensive response
        response = f"**{info.get('longName', ticker)} ({ticker})**\n\n"
        
        # Price information
        current_price = current_data['Close'].iloc[-1]
        open_price = current_data['Open'].iloc[-1]
        high = current_data['High'].iloc[-1]
        low = current_data['Low'].iloc[-1]
        volume = current_data['Volume'].iloc[-1]
        
        response += f"**Current Price:** ${current_price:.2f}\n"
        response += f"**Open:** ${open_price:.2f}\n"
        response += f"**Day High:** ${high:.2f}\n"
        response += f"**Day Low:** ${low:.2f}\n"
        response += f"**Volume:** {volume:,.0f}\n\n"
        
        # Company metrics
        if 'marketCap' in info:
            response += f"**Market Cap:** ${info['marketCap']:,.0f}\n"
        
        if 'trailingPE' in info:
            response += f"**P/E Ratio:** {info['trailingPE']:.2f}\n"
        
        if 'dividendYield' in info and info['dividendYield']:
            response += f"**Dividend Yield:** {info['dividendYield']*100:.2f}%\n"
        
        if 'fiftyTwoWeekHigh' in info:
            response += f"**52 Week High:** ${info['fiftyTwoWeekHigh']:.2f}\n"
        
        if 'fiftyTwoWeekLow' in info:
            response += f"**52 Week Low:** ${info['fiftyTwoWeekLow']:.2f}\n"
        
        # Business summary (truncated)
        if 'longBusinessSummary' in info:
            summary = info['longBusinessSummary'][:300] + "..." if len(info['longBusinessSummary']) > 300 else info['longBusinessSummary']
            response += f"\n**Company Overview:**\n{summary}\n"
        
        return response
        
    except Exception as e:
        return f"Error fetching stock data: {str(e)}. Please verify the ticker symbol."


def extract_ticker(query: str) -> Optional[str]:
    """
    Extract stock ticker from query
    Handles both explicit tickers and company names
    """
    import re
    
    # Common ticker patterns (1-5 uppercase letters)
    ticker_match = re.search(r'\b([A-Z]{1,5})\b', query)
    if ticker_match:
        return ticker_match.group(1)
    
    # Common company name to ticker mapping
    company_tickers = {
        'apple': 'AAPL',
        'tesla': 'TSLA',
        'microsoft': 'MSFT',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'meta': 'META',
        'facebook': 'META',
        'nvidia': 'NVDA',
        'amd': 'AMD',
        'intel': 'INTC',
        'netflix': 'NFLX',
        'disney': 'DIS',
        'walmart': 'WMT',
        'visa': 'V',
        'mastercard': 'MA',
        'jpmorgan': 'JPM',
        'bank of america': 'BAC',
        'berkshire': 'BRK-B',
        'coca cola': 'KO',
        'pepsi': 'PEP',
        'mcdonalds': 'MCD',
        'nike': 'NKE',
        'starbucks': 'SBUX'
    }
    
    query_lower = query.lower()
    for company, ticker in company_tickers.items():
        if company in query_lower:
            return ticker
    
    return None


# Enhanced Tavily tool with better configuration
tavilytool = TavilySearch(
    max_results=config["tools"]["tavily"]["max_results"],
    search_depth="advanced",
    include_answer=True,
    include_domains=["nseindia.com", "bseindia.com", "sebi.gov.in", "economictimes.indiatimes.com", 
 "livemint.com", "moneycontrol.com", "business-standard.com", "financialexpress.com",
 "screener.in", "in.investing.com", "bloomberg.com", "reuters.com"],  # Can specify trusted domains   # Can exclude unreliable sources
    topic="finance",
    api_key=os.environ.get("TAVILY_API_KEY")
)

# Legacy Polygon tool (keep for backwards compatibility)
from langchain_community.tools.polygon.financials import PolygonFinancials
from langchain_community.utilities.polygon import PolygonAPIWrapper

try:
    api_wrapper = PolygonAPIWrapper()
    financials_tool = PolygonFinancials(api_wrapper=api_wrapper)
except:
    print("Warning: Polygon API not configured. Using Yahoo Finance instead.")
    financials_tool = None