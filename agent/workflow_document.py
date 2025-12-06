"""
Document RAG Workflow - Two-LLM Architecture
- Gemini Flash: Multimodal document analysis (tables, charts, text)
- Groq: Fast conversational response generation with source citation
- Enhanced logging and exception handling
- STM + LTM session management
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from typing import List, Dict, Optional
import os
import sys
from datetime import datetime

# Internal imports
from utils.model_loaders import ModelLoader
from utils.memory_manager import HybridMemoryManager
from utils.cache_manager import get_cache_manager
from custom_logging.my_logger import logger, log_execution_time
from exception.exceptions import WorkflowException, VectorStoreException

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone import Pinecone
#from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chains.llm_requests import LLMChain

class DocumentState(TypedDict):
    """Enhanced state for two-LLM workflow"""
    messages: Annotated[list, add_messages]
    doc_id: str
    #session_id: str
    user_id: str
    retrieved_chunks: list
    sources: list
    gemini_analysis: str  # NEW: Gemini's multimodal analysis
    summary: str


class DocumentRAGWorkflowBuilder:
    """
    Two-LLM Document RAG Workflow:
    1. Retrieve relevant chunks from Pinecone
    2. Gemini Flash analyzes multimodal content (text/tables/charts)
    3. Groq generates conversational response with citations
    4. STM/LTM session management
    """
    
    def __init__(self):
        """Initialize two-LLM workflow"""
        try:
            logger.info("[DOCUMENT WORKFLOW] Initializing Two-LLM architecture...")
            
            # Model loader
            self.model_loader = ModelLoader()
            
            # LLM 1: Gemini Flash for multimodal analysis
            self.gemini_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.getenv('GOOGLE_API_KEY'),
                temperature=0.1
            )
            logger.info("[DOCUMENT WORKFLOW] Gemini Flash loaded for multimodal analysis")
            
            # LLM 2: Groq for fast conversational response
            self.groq_llm = self.model_loader.load_llm()  # Loads Groq from config
            logger.info("[DOCUMENT WORKFLOW] Groq loaded for response generation")
            
            # Embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            
            # Setup components
            self._setup_pinecone()
            self._setup_memory_manager()
            
            # Cache manager
            self.cache_manager = get_cache_manager(ttl_seconds=1800)
            
            # Prompts
            self._setup_gemini_prompt()
            self._setup_groq_prompt()
            self._setup_summary_chain()
            
            self.graph = None
            
            logger.info("[DOCUMENT WORKFLOW] Two-LLM initialization complete")
            
        except Exception as e:
            logger.error(f"[DOCUMENT WORKFLOW] Initialization failed: {e}")
            raise WorkflowException(
                "Failed to initialize document workflow",
                sys,
                component="initialization"
            )
    
    def _setup_pinecone(self):
        """Setup Pinecone vector store"""
        try:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found in environment")
            
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index = self.pc.Index("trading-bot")
            
            logger.info("[DOCUMENT WORKFLOW] Pinecone connected successfully")
            
        except Exception as e:
            logger.error(f"[PINECONE] Setup failed: {e}")
            raise VectorStoreException(
                "Failed to setup Pinecone",
                sys,
                service="pinecone"
            )
    
    def _setup_memory_manager(self):
        """Setup hybrid memory manager (Redis + Weaviate)"""
        try:
            self.memory_manager = HybridMemoryManager(
                redis_host=os.getenv('REDIS_HOST', 'localhost'),
                redis_port=int(os.getenv('REDIS_PORT', 6380)),
                weaviate_url=os.getenv('WEAVIATE_URL', 'http://localhost:8080'),
                weaviate_api_key=os.getenv('WEAVIATE_API_KEY'),
                session_ttl_hours=48
            )
            logger.info("[MEMORY] Hybrid memory manager initialized (Redis + Weaviate)")
            
        except Exception as e:
            logger.error(f"[MEMORY] Setup failed: {e}")
            raise WorkflowException(
                "Failed to setup memory manager",
                sys,
                component="memory_manager"
            )
    
    def _setup_gemini_prompt(self):
        """Setup Gemini Flash prompt for multimodal analysis"""
        self.gemini_system_prompt = """You are a **Multimodal Document Analyzer** powered by Gemini Flash.

**YOUR TASK:**
Analyze the provided document chunks (text, tables, or charts) and extract precise information to answer the user's query.

**ANALYSIS INSTRUCTIONS:**

1. **For TEXT chunks:**
   - Extract relevant sentences and paragraphs
   - Preserve exact quotes when important
   - Note the page number

2. **For TABLE chunks:**
   - Parse the table structure carefully
   - Extract specific data points requested
   - Understand relationships between columns/rows
   - Note: Table ID and Page Number

3. **For CHART chunks:**
   - Analyze chart descriptions and data
   - Extract trends, values, and insights
   - Note: Chart ID and Page Number

**OUTPUT FORMAT:**
Return a structured analysis in this format:

```
ANALYSIS:
[Your detailed analysis here]

EXTRACTED DATA:
- [Key point 1 from source X]
- [Key point 2 from source Y]
...

SOURCES USED:
- Page X (Text)
- Table T1 on Page Y
- Chart C2 on Page Z
```

**CRITICAL RULES:**
- Use ONLY the provided chunks - no external knowledge
- Be precise with numbers and data
- Clearly cite sources for each claim
- If information is missing, state it explicitly

---

**USER QUERY:**
{query}

**RETRIEVED CHUNKS:**
{chunks}
"""
    
    def _setup_groq_prompt(self):
        """Setup Groq prompt for conversational response"""
        self.groq_system_prompt = """You are a **Professional Document Assistant** powered by Groq for fast, accurate responses.

**YOUR ROLE:**
Generate a natural, conversational response based on the multimodal analysis provided by our document analyzer.

**RESPONSE GUIDELINES:**

1. **Conversational Tone:** Write naturally, as if explaining to a colleague
2. **Source Citation:** ALWAYS cite sources using this format:
   - Tables: "According to Table T1 on Page 3, ..."
   - Charts: "As shown in Chart C2 on Page 5, ..."
   - Text: "Based on Page 2, ..."
3. **Accuracy:** Use exact data from the analysis
4. **Clarity:** Structure your response logically with proper formatting
5. **Completeness:** Answer the full query comprehensively

**CONVERSATION CONTEXT:**
{conversation_history}

**PREVIOUS INTERACTIONS WITH THIS DOCUMENT:**
{long_term_context}

**MULTIMODAL ANALYSIS:**
{gemini_analysis}

**USER QUERY:**
{query}

**INSTRUCTIONS:**
- Synthesize the analysis into a clear, helpful response
- Cite ALL sources for factual claims
- Use bullet points or numbering for clarity when appropriate
- If the analysis indicates missing information, acknowledge it
"""
    
    def _setup_summary_chain(self):
        """Setup summarization chain for LTM"""
        try:
            summary_prompt = PromptTemplate.from_template(
                """Summarize this document conversation concisely (under 100 words):
                
                Current Summary: {current_summary}
                New Messages: {new_messages}
                
                Focus on:
                - Key questions asked
                - Main topics discussed
                - Important findings from the document
                
                Return ONLY the updated summary.
                """
            )
            self.summary_chain = LLMChain(llm=self.groq_llm, prompt=summary_prompt)
            logger.info("[SUMMARY] Chain initialized with Groq")
            
        except Exception as e:
            logger.error(f"[SUMMARY] Setup failed: {e}")
            raise WorkflowException(
                "Failed to setup summary chain",
                sys,
                component="summary_chain"
            )
    
    @log_execution_time
    def _retrieval_node(self, state: DocumentState) -> DocumentState:
        """
        Node 1: Retrieve relevant chunks from Pinecone
        """
        try:
            messages = state["messages"]
            doc_id = state.get("doc_id", "")
            
            last_message = messages[-1] if messages else None
            
            if not last_message or not hasattr(last_message, 'content'):
                logger.warning("[RETRIEVAL] No valid message found")
                return state
            
            query = last_message.content
            logger.info(f"[RETRIEVAL] Doc: {doc_id} | Query: {query[:50]}...")
            
            # Create vector store with doc_id filter
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="documents"
            )
            
            # Retrieve top-k similar chunks
            results = vector_store.similarity_search(
                query=query,
                k=5,
                filter={"doc_id": doc_id}
            )
            
            chunks = []
            sources = []
            
            for doc in results:
                chunks.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'type': doc.metadata.get('type', 'text')
                })
                
                # Extract source information
                metadata = doc.metadata
                if metadata.get('type') == 'table':
                    sources.append({
                        'type': 'table',
                        'page': metadata.get('page_num'),
                        'table_id': metadata.get('table_id'),
                        'table_name': metadata.get('table_name', f"Table {metadata.get('table_id')}")
                    })
                elif metadata.get('type') == 'chart':
                    sources.append({
                        'type': 'chart',
                        'page': metadata.get('page_num'),
                        'chart_id': metadata.get('chart_id'),
                        'chart_name': metadata.get('chart_name', f"Chart {metadata.get('chart_id')}")
                    })
                elif metadata.get('page_num'):
                    sources.append({
                        'type': 'text',
                        'page': metadata.get('page_num')
                    })
            
            logger.info(f"[RETRIEVAL] Retrieved {len(chunks)} chunks | Sources: {len(sources)}")
            
            return {
                **state,
                "retrieved_chunks": chunks,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"[RETRIEVAL] Error: {e}")
            raise VectorStoreException(
                "Retrieval failed",
                sys,
                doc_id=doc_id,
                query=query[:100] if 'query' in locals() else "empty"
            )
    
    @log_execution_time
    def _gemini_analysis_node(self, state: DocumentState) -> Dict:
        """
        Node 2: Gemini Flash analyzes multimodal content
        """
        try:
            messages = state["messages"]
            retrieved_chunks = state.get("retrieved_chunks", [])
            
            last_message = messages[-1] if messages else None
            if not last_message:
                return {"gemini_analysis": "No query provided"}
            
            query = last_message.content
            logger.info(f"[GEMINI] Starting multimodal analysis...")
            
            # Check cache
            cache_key = self.cache_manager._generate_key(
                query, str(retrieved_chunks), prefix="gemini_analysis"
            )
            cached_analysis = self.cache_manager.get(cache_key)
            
            if cached_analysis:
                logger.info("[GEMINI] Using cached analysis")
                return {"gemini_analysis": cached_analysis}
            
            # Format chunks for Gemini
            formatted_chunks = self._format_chunks_for_gemini(retrieved_chunks)
            
            # Build Gemini prompt
            gemini_prompt = self.gemini_system_prompt.format(
                query=query,
                chunks=formatted_chunks
            )
            
            # Invoke Gemini Flash
            gemini_response = self.gemini_llm.invoke([
                SystemMessage(content=gemini_prompt)
            ])
            
            analysis = gemini_response.content if hasattr(gemini_response, 'content') else str(gemini_response)
            
            # Cache the analysis
            self.cache_manager.set(cache_key, analysis)
            
            logger.info(f"[GEMINI] Analysis complete | Length: {len(analysis)} chars")
            
            return {"gemini_analysis": analysis}
            
        except Exception as e:
            logger.error(f"[GEMINI] Analysis failed: {e}")
            return {"gemini_analysis": f"Analysis error: {str(e)}"}
    
    def _format_chunks_for_gemini(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks for Gemini analysis"""
        formatted = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']
            content = chunk['content']
            chunk_type = chunk.get('type', 'text')
            
            if chunk_type == 'table':
                formatted.append(
                    f"--- CHUNK {i}: TABLE ---\n"
                    f"Table ID: {metadata.get('table_id', 'Unknown')}\n"
                    f"Page: {metadata.get('page_num', 'Unknown')}\n"
                    f"Content:\n{content}\n"
                )
            elif chunk_type == 'chart':
                formatted.append(
                    f"--- CHUNK {i}: CHART ---\n"
                    f"Chart ID: {metadata.get('chart_id', 'Unknown')}\n"
                    f"Page: {metadata.get('page_num', 'Unknown')}\n"
                    f"Description:\n{content}\n"
                )
            else:
                formatted.append(
                    f"--- CHUNK {i}: TEXT ---\n"
                    f"Page: {metadata.get('page_num', 'Unknown')}\n"
                    f"Content:\n{content}\n"
                )
        
        return "\n".join(formatted)
    
    @log_execution_time
    def _groq_response_node(self, state: DocumentState) -> Dict:
        """
        Node 3: Groq generates conversational response with citations
        """
        try:
            messages = state["messages"]
            #session_id = state.get("session_id", "default")
            user_id = state.get("user_id", "anonymous")
            doc_id = state.get("doc_id", "")
            gemini_analysis = state.get("gemini_analysis", "No analysis available")
            
            last_message = messages[-1] if messages else None
            if not last_message:
                return {"messages": [AIMessage(content="No query provided.")]}
            
            query = last_message.content
            logger.info(f"[GROQ] Generating conversational response...")
            
            # Check cache
            cache_key = self.cache_manager._generate_key(
                doc_id, query, gemini_analysis[:100], prefix="groq_response"
            )
            cached_response = self.cache_manager.get(cache_key)
            
            if cached_response:
                logger.info("[GROQ] Using cached response")
                return {"messages": [cached_response]}
            
            # Get conversation history (STM)
            redis_history = self.memory_manager.get_short_term_memory(doc_id, limit=5)
            conversation_history = "\n".join([
                f"{type(msg).__name__}: {msg.content[:100]}..."
                for msg in redis_history[-3:]
            ])
            
            # Get long-term context (LTM)
            long_term_context = self._get_document_ltm(doc_id, user_id)
            
            # Build Groq prompt
            groq_prompt = self.groq_system_prompt.format(
                conversation_history=conversation_history or "No previous conversation",
                long_term_context=long_term_context,
                gemini_analysis=gemini_analysis,
                query=query
            )
            
            # Invoke Groq
            groq_response = self.groq_llm.invoke([
                SystemMessage(content=groq_prompt)
            ])
            
            # Cache response
            self.cache_manager.set(cache_key, groq_response)
            
            # Store in STM
            if hasattr(groq_response, 'content'):
                self.memory_manager.store_message(doc_id, "assistant", groq_response.content)
            
            logger.info(f"[GROQ] Response generated | Length: {len(groq_response.content) if hasattr(groq_response, 'content') else 0}")
            
            return {"messages": [groq_response]}
            
        except Exception as e:
            logger.error(f"[GROQ] Response generation failed: {e}")
            error_msg = AIMessage(
                content="I encountered an error generating the response. Please try again."
            )
            return {"messages": [error_msg]}
    
    def _get_document_ltm(self, doc_id: str, user_id: str) -> str:
        """Retrieve long-term memory for document"""
        try:
            collection = self.memory_manager.weaviate_manager.client.collections.get("DocumentQA")
            
            from weaviate.classes.query import Filter
            response = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=5
            )
            
            if response.objects:
                context = "Previous Q&A about this document:\n"
                for item in response.objects:
                    q = item.properties.get('question', '')[:60]
                    a = item.properties.get('answer', '')[:120]
                    context += f"Q: {q}...\nA: {a}...\n\n"
                return context
        except Exception as e:
            logger.warning(f"[LTM] Could not retrieve context: {e}")
        
        return "No previous interactions with this document."
    
    @log_execution_time
    def _summarization_node(self, state: DocumentState) -> Dict:
        """
        Node 4: Summarize and archive to LTM
        """
        try:
            messages = state["messages"]
            #session_id = state.get("session_id", "default")
            user_id = state.get("user_id", "anonymous")
            doc_id = state.get("doc_id", "")
            
            current_summary = state.get("summary", "New document conversation")
            
            conversational_messages = [
                m for m in messages 
                if isinstance(m, (HumanMessage, AIMessage))
            ]
            
            new_messages = conversational_messages[-5:]
            
            if not new_messages:
                return state
            
            formatted = "\n".join([
                f"{type(m).__name__}: {m.content[:150]}" 
                for m in new_messages
            ])
            
            # Generate summary
            new_summary_result = self.summary_chain.invoke({
                "current_summary": current_summary,
                "new_messages": formatted
            })
            
            new_summary = new_summary_result['text'].strip()
            logger.info(f"[SUMMARY] Updated for session: {doc_id}")
            
            # Archive to LTM if substantial (10+ messages)
            if len(conversational_messages) >= 10:
                logger.info(f"[ARCHIVING] Archiving session {doc_id} to LTM")
                self.memory_manager.archive_conversation(
                    doc_id=doc_id,
                    summary=new_summary,
                    key_topics=['document', 'analysis', doc_id],
                    user_preferences={'doc_id': doc_id},
                    user_id=user_id
                )
            
            return {"summary": new_summary}
            
        except Exception as e:
            logger.error(f"[SUMMARY] Error: {e}")
            return state
    
    def build(self):
        """Build two-LLM workflow graph"""
        try:
            logger.info("[DOCUMENT WORKFLOW] Building two-LLM graph...")
            
            graph_builder = StateGraph(DocumentState)
            
            # Add nodes in order
            graph_builder.add_node("retrieval", self._retrieval_node)
            graph_builder.add_node("gemini_analysis", self._gemini_analysis_node)
            graph_builder.add_node("groq_response", self._groq_response_node)
            graph_builder.add_node("summarizer", self._summarization_node)
            
            # Define workflow edges
            graph_builder.add_edge(START, "retrieval")
            graph_builder.add_edge("retrieval", "gemini_analysis")
            graph_builder.add_edge("gemini_analysis", "groq_response")
            graph_builder.add_edge("groq_response", "summarizer")
            graph_builder.add_edge("summarizer", END)
            
            self.graph = graph_builder.compile()
            
            logger.info("[DOCUMENT WORKFLOW] Two-LLM graph built successfully")
            
        except Exception as e:
            logger.error(f"[DOCUMENT WORKFLOW] Graph building failed: {e}")
            raise WorkflowException(
                "Failed to build workflow graph",
                sys,
                component="graph_builder"
            )
    
    def get_graph(self):
        """Get compiled graph"""
        if not self.graph:
            raise WorkflowException(
                "Graph not built. Call build() first.",
                sys,
                component="graph"
            )
        return self.graph
    
    def clear_cache(self):
        """Clear all document caches"""
        count = self.cache_manager.clear(prefix="gemini_analysis")
        count += self.cache_manager.clear(prefix="groq_response")
        logger.info(f"[CACHE] Cleared {count} document cache entries")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return self.cache_manager.get_stats()
    
    def close(self):
        """Cleanup resources"""
        try:
            logger.info("[DOCUMENT WORKFLOW] Closing connections...")
            self.memory_manager.close()
            logger.info("[DOCUMENT WORKFLOW] Connections closed")
        except Exception as e:
            logger.error(f"[DOCUMENT WORKFLOW] Error during cleanup: {e}")


# Export
__all__ = ['DocumentRAGWorkflowBuilder']