from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any , Optional
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class EnhancedMultiModalRetriever:
    """
    Enhanced retriever with document-specific filtering
    Supports text, images, and tables with document isolation
    """
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.config = load_config()
        
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index_name = self.config["vector_db"]["index_name"]
        
        pc = Pinecone(api_key=pinecone_api_key)
        self.index = pc.Index(index_name)
    
    def search_document(
        self,
        query: str,
        doc_id: Optional[str] = None,
        search_type: str = "all",
        k: int = 5
    ) -> str:
        """
        Search multimodal content with optional document filtering
        
        Args:
            query: Search query
            doc_id: Optional document ID to restrict search
            search_type: 'text', 'images', 'tables', or 'all'
            k: Number of results per type
            
        Returns:
            Formatted search results
        """
        results = []
        
        # Build metadata filter for document
        filter_dict = {"doc_id": doc_id} if doc_id else None
        
        # Search text namespace
        if search_type in ["all", "text"]:
            text_results = self._search_namespace(
                "text", query, k, filter_dict
            )
            results.extend([{
                'type': 'text',
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in text_results])
        
        # Search images namespace
        if search_type in ["all", "images"]:
            image_results = self._search_namespace(
                "images", query, k, filter_dict
            )
            results.extend([{
                'type': 'image',
                'caption': doc.page_content,
                'metadata': doc.metadata
            } for doc in image_results])
        
        # Search tables namespace
        if search_type in ["all", "tables"]:
            table_results = self._search_namespace(
                "tables", query, k, filter_dict
            )
            results.extend([{
                'type': 'table',
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in table_results])
        
        if not results:
            if doc_id:
                return f"No relevant information found in this document for: {query}"
            return "No relevant information found in the knowledge base."
        
        return self._format_results(results)
    
    def _search_namespace(
        self,
        namespace: str,
        query: str,
        k: int,
        filter_dict: Optional[Dict] = None
    ) -> List[Any]:
        """Search specific namespace with optional filtering"""
        try:
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace=namespace
            )
            
            # Use similarity search with metadata filter
            if filter_dict:
                results = vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Namespace search failed ({namespace}): {e}")
            return []
    
    def _format_results(self, results: List[Dict]) -> str:
        """Format multimodal results for LLM consumption"""
        formatted = []
        
        for r in results:
            content_type = r['type']
            metadata = r.get('metadata', {})
            
            if content_type == 'text':
                page = metadata.get('page', 'unknown')
                source = metadata.get('source', 'document')
                formatted.append(
                    f"[TEXT from {source}, page {page}]\n{r['content']}\n"
                )
            
            elif content_type == 'image':
                page = metadata.get('page_num', 'unknown')
                caption = r.get('caption', 'Image')
                formatted.append(
                    f"[IMAGE on page {page}]\nCaption: {caption}\n"
                )
            
            elif content_type == 'table':
                page = metadata.get('page', 'unknown')
                formatted.append(
                    f"[TABLE from page {page}]\n{r['content']}\n"
                )
        
        return "\n".join(formatted)
    
    def search_by_type(
        self,
        query: str,
        content_type: str,
        doc_id: Optional[str] = None,
        k: int = 3
    ) -> str:
        """
        Search for specific content type
        
        Args:
            query: Search query
            content_type: 'text', 'image', or 'table'
            doc_id: Optional document filter
            k: Number of results
        """
        return self.search_document(query, doc_id, content_type, k)


# Global retriever instance
enhanced_retriever = EnhancedMultiModalRetriever()


@tool
def document_multimodal_retriever_tool(query: str, doc_id: str = None) -> str:
    """
    Search multimodal knowledge base (text, images, tables) for document-specific queries.
    
    USE THIS TOOL for document-based questions about:
    - Content from uploaded PDFs or documents
    - Charts, images, or visual elements in documents
    - Tables or structured data in documents
    - Historical information stored in documents
    
    Args:
        query: The search query
        doc_id: Optional document ID to restrict search to specific document
        
    Returns:
        Relevant content from text, images, and tables
    """
    try:
        return enhanced_retriever.search_document(
            query=query,
            doc_id=doc_id,
            search_type="all",
            k=5
        )
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"


@tool
def find_document_images(query: str, doc_id: str = None) -> str:
    """
    Search specifically for images/charts in documents.
    
    Args:
        query: What to look for in images
        doc_id: Optional document ID
    """
    try:
        return enhanced_retriever.search_by_type(
            query=query,
            content_type="images",
            doc_id=doc_id,
            k=3
        )
    except Exception as e:
        return f"Error searching images: {str(e)}"


@tool
def find_document_tables(query: str, doc_id: str = None) -> str:
    """
    Search specifically for tables in documents.
    
    Args:
        query: What to look for in tables
        doc_id: Optional document ID
    """
    try:
        return enhanced_retriever.search_by_type(
            query=query,
            content_type="tables",
            doc_id=doc_id,
            k=3
        )
    except Exception as e:
        return f"Error searching tables: {str(e)}"

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
    'compare_stocks_tool',
    'get_stock_price_tool',
    'document_multimodal_retriever_tool',
    'find_document_images',
    'find_document_tables',
    'EnhancedMultiModalRetriever'
]
