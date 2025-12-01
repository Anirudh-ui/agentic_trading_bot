from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class MultiModalRetriever:
    """Enhanced retriever that searches across text, images, and tables"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.config = load_config()
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")#self.config["vector_db"]["api_key"]
        index_name = self.config["vector_db"]["index_name"]
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
    
    def search(self, query: str, search_type: str = "all", k: int = 5) -> str:
        """
        Search across multi-modal content
        
        Args:
            query: Search query
            search_type: 'text', 'images', 'tables', or 'all'
            k: Number of results to return
        """
        results = []
        
        if search_type in ["all", "text"]:
            text_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="text"
            )
            text_results = text_store.similarity_search(query, k=k)
            results.extend([{
                'type': 'text',
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in text_results])
        
        if search_type in ["all", "images"]:
            image_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="images"
            )
            image_results = image_store.similarity_search(query, k=k)
            results.extend([{
                'type': 'image',
                'caption': doc.page_content,
                'metadata': doc.metadata
            } for doc in image_results])
        
        if search_type in ["all", "tables"]:
            table_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="tables"
            )
            table_results = table_store.similarity_search(query, k=k)
            results.extend([{
                'type': 'table',
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in table_results])
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        # Format results
        formatted = []
        for r in results:
            if r['type'] == 'text':
                formatted.append(f"[TEXT] {r['content']}")
            elif r['type'] == 'image':
                formatted.append(f"[IMAGE] {r['caption']}")
            elif r['type'] == 'table':
                formatted.append(f"[TABLE] {r['content']}")
        
        return "\n\n".join(formatted)


# Initialize global retriever
multimodal_retriever = MultiModalRetriever()


@tool
def multimodal_retriever_tool(query: str) -> str:
    """
    Search the multi-modal knowledge base (text, images, tables) for trading information.
    Use this for questions about:
    - Trading strategies and concepts from documents
    - Charts and visual analysis from uploaded PDFs
    - Financial tables and data from documents
    - Historical trading information
    
    Args:
        query: The search query
    """
    try:
        return multimodal_retriever.search(query, search_type="all", k=5)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


class StockComparisonEngine:
    """Engine for comprehensive stock comparisons"""
    
    @staticmethod
    def get_stock_data(ticker: str, period: str = "1mo") -> Dict[str, Any]:
        """Fetch comprehensive stock data"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist = stock.history(period=period)
            
            # Get stock info
            info = stock.info
            
            # Calculate key metrics
            current_price = hist['Close'].iloc[-1] if len(hist) > 0 else None
            price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[0] if len(hist) > 1 else 0
            percent_change = (price_change / hist['Close'].iloc[0] * 100) if len(hist) > 1 else 0
            
            avg_volume = hist['Volume'].mean() if len(hist) > 0 else None
            volatility = hist['Close'].pct_change().std() * 100 if len(hist) > 1 else None
            
            return {
                'ticker': ticker,
                'current_price': round(current_price, 2) if current_price else None,
                'price_change': round(price_change, 2),
                'percent_change': round(percent_change, 2),
                'avg_volume': int(avg_volume) if avg_volume else None,
                'volatility': round(volatility, 2) if volatility else None,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                '52_week_high': info.get('fiftyTwoWeekHigh'),
                '52_week_low': info.get('fiftyTwoWeekLow'),
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}
    
    @staticmethod
    def compare_stocks(tickers: List[str], period: str = "1mo") -> str:
        """
        Compare multiple stocks comprehensively
        
        Args:
            tickers: List of stock ticker symbols
            period: Time period for comparison (1d, 5d, 1mo, 3mo, 6mo, 1y)
        """
        if len(tickers) < 2:
            return "Please provide at least 2 stock tickers for comparison."
        
        print(f"[COMPARISON] Comparing {len(tickers)} stocks: {', '.join(tickers)}")
        
        # Fetch data for all stocks
        stocks_data = []
        for ticker in tickers:
            data = StockComparisonEngine.get_stock_data(ticker, period)
            stocks_data.append(data)
        
        # Build comparison report
        report = f"# Stock Comparison Report ({period})\n\n"
        report += f"**Stocks Analyzed:** {', '.join(tickers)}\n"
        report += f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Price comparison
        report += "## Current Price Comparison\n\n"
        for data in stocks_data:
            if 'error' not in data and data['current_price']:
                report += f"- **{data['company_name']} ({data['ticker']})**: ${data['current_price']}\n"
        
        # Performance comparison
        report += "\n## Performance Comparison\n\n"
        for data in stocks_data:
            if 'error' not in data:
                change_indicator = "🟢" if data['percent_change'] > 0 else "🔴"
                report += f"- **{data['ticker']}**: {change_indicator} {data['percent_change']}% "
                report += f"(${data['price_change']})\n"
        
        # Best/Worst performers
        valid_stocks = [d for d in stocks_data if 'error' not in d and d['percent_change'] is not None]
        if valid_stocks:
            best = max(valid_stocks, key=lambda x: x['percent_change'])
            worst = min(valid_stocks, key=lambda x: x['percent_change'])
            
            report += f"\n**Best Performer:** {best['ticker']} (+{best['percent_change']}%)\n"
            report += f"**Worst Performer:** {worst['ticker']} ({worst['percent_change']}%)\n"
        
        # Volatility comparison
        report += "\n## Volatility Analysis\n\n"
        for data in stocks_data:
            if 'error' not in data and data['volatility']:
                volatility_level = "High" if data['volatility'] > 3 else "Medium" if data['volatility'] > 1.5 else "Low"
                report += f"- **{data['ticker']}**: {data['volatility']}% ({volatility_level})\n"
        
        # Fundamental comparison
        report += "\n## Fundamental Comparison\n\n"
        
        # Create comparison table
        comparison_df = pd.DataFrame([
            {
                'Ticker': d['ticker'],
                'Market Cap': f"${d['market_cap']/1e9:.2f}B" if d.get('market_cap') else "N/A",
                'P/E Ratio': f"{d['pe_ratio']:.2f}" if d.get('pe_ratio') else "N/A",
                'Dividend Yield': f"{d['dividend_yield']*100:.2f}%" if d.get('dividend_yield') else "N/A",
                'Sector': d.get('sector', 'N/A')
            }
            for d in stocks_data if 'error' not in d
        ])
        
        report += comparison_df.to_markdown(index=False)
        
        # 52-week range comparison
        report += "\n\n## 52-Week Range\n\n"
        for data in stocks_data:
            if 'error' not in data and data.get('52_week_high') and data.get('52_week_low'):
                current = data['current_price']
                high = data['52_week_high']
                low = data['52_week_low']
                position = ((current - low) / (high - low)) * 100 if high != low else 50
                
                report += f"- **{data['ticker']}**: ${low:.2f} - ${high:.2f} "
                report += f"(Currently at {position:.0f}% of range)\n"
        
        # Investment recommendations
        report += "\n## Key Insights\n\n"
        
        for data in stocks_data:
            if 'error' not in data:
                insights = []
                
                if data['percent_change'] > 5:
                    insights.append(f"Strong upward momentum (+{data['percent_change']}%)")
                elif data['percent_change'] < -5:
                    insights.append(f"Significant decline ({data['percent_change']}%)")
                
                if data.get('pe_ratio') and data['pe_ratio'] < 15:
                    insights.append("Potentially undervalued (Low P/E)")
                elif data.get('pe_ratio') and data['pe_ratio'] > 30:
                    insights.append("High P/E ratio (growth premium or overvalued)")
                
                if data.get('dividend_yield') and data['dividend_yield'] > 0.03:
                    insights.append(f"Good dividend yield ({data['dividend_yield']*100:.2f}%)")
                
                if insights:
                    report += f"**{data['ticker']}:** {', '.join(insights)}\n"
        
        # Errors (if any)
        errors = [d for d in stocks_data if 'error' in d]
        if errors:
            report += "\n## Errors\n\n"
            for err in errors:
                report += f"- **{err['ticker']}**: {err['error']}\n"
        
        return report


@tool
def compare_stocks_tool(tickers_json: str) -> str:
    """
    Compare multiple stocks comprehensively (prices, performance, fundamentals).
    
    Use this tool when the user asks to compare stocks, e.g.:
    - "Compare Apple and Tesla stock"
    - "How do AAPL, MSFT, and GOOGL compare?"
    - "Compare the performance of Amazon and Netflix"
    
    Args:
        tickers_json: JSON string with format: {"tickers": ["AAPL", "TSLA"], "period": "1mo"}
                     Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y
    
    Returns:
        Comprehensive comparison report
    """
    try:
        data = json.loads(tickers_json)
        tickers = data.get('tickers', [])
        period = data.get('period', '1mo')
        
        if not tickers or len(tickers) < 2:
            return "Error: Please provide at least 2 stock tickers for comparison."
        
        return StockComparisonEngine.compare_stocks(tickers, period)
    
    except json.JSONDecodeError:
        return "Error: Invalid JSON format. Use: {\"tickers\": [\"AAPL\", \"TSLA\"], \"period\": \"1mo\"}"
    except Exception as e:
        return f"Error comparing stocks: {str(e)}"


@tool
def get_stock_price_tool(ticker: str) -> str:
    """
    Get current stock price and basic information for a single stock.
    
    Use this for simple price queries like:
    - "What's the price of Apple stock?"
    - "Current price of TSLA"
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, MSFT)
    """
    try:
        data = StockComparisonEngine.get_stock_data(ticker, period="5d")
        
        if 'error' in data:
            return f"Error fetching {ticker}: {data['error']}"
        
        response = f"**{data['company_name']} ({data['ticker']})**\n\n"
        response += f"Current Price: ${data['current_price']}\n"
        response += f"Change: {'+' if data['price_change'] > 0 else ''}{data['price_change']} "
        response += f"({'+' if data['percent_change'] > 0 else ''}{data['percent_change']}%)\n"
        
        if data.get('market_cap'):
            response += f"Market Cap: ${data['market_cap']/1e9:.2f}B\n"
        
        if data.get('pe_ratio'):
            response += f"P/E Ratio: {data['pe_ratio']:.2f}\n"
        
        if data.get('sector'):
            response += f"Sector: {data['sector']}\n"
        
        return response
    
    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


# Export all tools
__all__ = [
    'multimodal_retriever_tool',
    'compare_stocks_tool',
    'get_stock_price_tool'
]
