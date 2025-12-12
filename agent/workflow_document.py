"""
DOCUMENT RAG WORKFLOW (Final Stable Version)
--------------------------------------------
Two-LLM Architecture:
 - Gemini 2.5 Flash Lite → Deep chunk understanding
 - Groq Qwen-32B → Final answer with citations
 - Pinecone → Vector search
 - HybridMemoryManager → STM (Redis) + LTM (Weaviate)

This version:
 - Uses Redis summary first, then Weaviate summary
 - Clean separation of STM/LTM (no .ltm usage)
 - Full intent routing integrated
 - Correct summarizer node
 - Proper citation formatting
"""

import os
import sys
from typing import Dict, List
from datetime import datetime

from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from vertexai.generative_models import GenerativeModel

from utils.model_loaders import ModelLoader
from utils.memory_manager import HybridMemoryManager
from utils.cache_manager import get_cache_manager

from custom_logging.my_logger import logger, log_execution_time
from exception.exceptions import WorkflowException, VectorStoreException
from utils.document_intent_classifier import DocumentIntentClassifier

# ================================================================
# WORKFLOW STATE
# ================================================================

class DocumentState(TypedDict):
    messages: Annotated[list, add_messages]
    doc_id: str
    user_id: str
    retrieved_chunks: List[Dict]
    sources: List[Dict]
    gemini_analysis: str
    summary: str


# ================================================================
# DOCUMENT WORKFLOW BUILDER
# ================================================================

class DocumentRAGWorkflowBuilder:

    def __init__(self):
        try:
            logger.info("[WORKFLOW] Initializing Document Workflow...")

            # ---------------------
            # Load LLMs
            # ---------------------
            self.gemini_llm = GenerativeModel("gemini-2.5-flash-lite")
            self.model_loader = ModelLoader()
            self.groq_llm = self.model_loader.load_llm()

            # ---------------------
            # Pinecone setup
            # ---------------------
            self._setup_pinecone()

            # ---------------------
            # Memory manager (STM + LTM)
            # ---------------------
            self._setup_memory()

            # ---------------------
            # Intent classifier
            # ---------------------
            self.classifier = DocumentIntentClassifier()

            # ---------------------
            # Cache
            # ---------------------
            self.cache_manager = get_cache_manager(ttl_seconds=1800)

            # ---------------------
            # Prompts
            # ---------------------
            self._setup_prompts()
            self._setup_summary_prompt()

            self.graph = None
            logger.info("[WORKFLOW] Document workflow initialized successfully")

        except Exception as e:
            logger.error(f"[INIT ERR] {e}")
            raise WorkflowException("Document workflow initialization failed", sys)


    # ============================================================
    # PINECONE INIT
    # ============================================================

    def _setup_pinecone(self):
        try:
            key = os.getenv("PINECONE_API_KEY")
            if not key:
                raise ValueError("Missing PINECONE_API_KEY")

            self.pc = Pinecone(api_key=key)
            self.index = self.pc.Index("trading-bot")

            logger.info("[PINECONE] Connected OK")

        except Exception as e:
            logger.error(f"[PINECONE ERROR] {e}")
            raise VectorStoreException("Failed to initialize Pinecone", sys)


    # ============================================================
    # MEMORY INIT
    # ============================================================

    def _setup_memory(self):
        try:
            self.memory_manager = HybridMemoryManager(
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", 6380)),
                weaviate_url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
            )
            logger.info("[MEMORY] HybridMemoryManager ready")

        except Exception as e:
            logger.error(f"[MEMORY INIT ERROR] {e}")
            raise


    # ============================================================
    # PROMPTS
    # ============================================================

    def _setup_prompts(self):

        # ---------------- GENIMI PROMPT ----------------
        self.gemini_prompt = """
You are Gemini 2.5 Flash Lite.
Analyze ONLY the retrieved document chunks.
Extract:
 - Key facts
 - Definitions
 - Numerical values
 - Table or chart meaning
 - Important relationships
NEVER hallucinate.
"""

        # ---------------- GROQ PROMPT ----------------
        # Citations included
        self.groq_prompt = """
You are Groq (Qwen-32B). You produce the FINAL document-grounded answer.

RULES:
1. Use Gemini analysis as authoritative.
2. Use ONLY the retrieved chunks — do not hallucinate.
3. Cite sources using:
     (Page X), (Table T1), (Chart C3), (Image I2)
4. If the document does NOT contain the answer, say:
     "The document does not provide information about this."

CONVERSATION HISTORY:
{history}

GEMINI ANALYSIS:
{analysis}

DOCUMENT CHUNKS:
{chunks}

USER QUESTION:
{query}
"""


    # ============================================================
    # SUMMARY PROMPT (STM + LTM)
    # ============================================================

    def _setup_summary_prompt(self):
        from langchain_core.prompts import PromptTemplate

        self.summary_chain = (
            PromptTemplate.from_template(
                """
You update a rolling summary of a document conversation (<100 words).

Current Summary:
{current_summary}

Recent Messages:
{new_messages}

Return ONLY the updated summary.
"""
            ) | self.groq_llm
        )


    # ============================================================
    # RETRIEVAL NODE
    # ============================================================

    @log_execution_time
    def _retrieval_node(self, state: DocumentState):

        try:
            query = state["messages"][-1].content
            doc_id = state["doc_id"]

            logger.info(f"[RETRIEVAL] doc_id={doc_id} | query='{query}'")

            embed = self.model_loader.load_embeddings_vertex()

            vs = PineconeVectorStore(
                index=self.index,
                embedding=embed,
                namespace="documents"
            )

            # 🔥 NEW: Get chunks WITH similarity score
            results = vs.similarity_search_with_score(
                query=query,
                k=10,
                filter={"doc_id": doc_id}
            )

            chunks = []
            sources = []

            for i, (doc, score) in enumerate(results):
                meta = doc.metadata or {}

                chunks.append({
                    "content": doc.page_content,
                    "metadata": meta,
                    "score": float(score),     # 🔥 store score
                    "type": meta.get("type", "text")
                })

                sources.append({
                    "type": meta.get("type", "text"),
                    "page": meta.get("page_num"),
                    "id": meta.get("table_id") or meta.get("chart_id")
                })

            return {**state, "retrieved_chunks": chunks, "sources": sources}

        except Exception as e:
            logger.error(f"[RETRIEVAL ERR] {e}")
            return state


    # ============================================================
    # GEMINI ANALYSIS NODE
    # ============================================================

    @log_execution_time
    def _gemini_analysis_node(self, state: DocumentState):

        try:
            query = state["messages"][-1].content
            chunks = state["retrieved_chunks"]

            chunk_text = "\n\n".join(
                f"[CHUNK {i}] Page {c['metadata'].get('page_num')}:\n{c['content']}"
                for i, c in enumerate(chunks, 1)
            )

            full_prompt = (
                self.gemini_prompt
                + "\nQUESTION:\n" + query
                + "\n\nCHUNKS:\n" + chunk_text
            )

            resp = self.gemini_llm.generate_content(full_prompt)
            analysis = getattr(resp, "text", None) or "No analyzable content found."

            return {**state, "gemini_analysis": analysis}

        except Exception as e:
            logger.error(f"[GEMINI ERR] {e}")
            return {**state, "gemini_analysis": "Gemini analysis unavailable."}


    # ============================================================
    # GROQ ANSWER NODE (final answer generator)
    # ============================================================

    @log_execution_time
    def _groq_response_node(self, state: DocumentState):

        try:
            doc_id = state["doc_id"]
            query = state["messages"][-1].content.lower()

            # -------------------------------
            # 1) Get STM conversation history
            # -------------------------------
            history_events = self.memory_manager.get_short_term_memory(doc_id, limit=5)
            history_text = ""
            for m in history_events:
                history_text += f"{m['role'].upper()}: {m['content']}\n"

            # -------------------------------
            # 2) Smart Ranking of chunks
            # -------------------------------
            chunks = state.get("retrieved_chunks", [])

            def score_chunk(c):
                base = c.get("score", 1.0)

                bonus = 0

                # Priority 1 — tables & charts
                if c["type"] == "table":
                    bonus += 0.5
                if c["type"] == "chart":
                    bonus += 0.4

                # Priority 2 — keyword match in content
                content = c.get("content", "").lower()
                for word in query.split():
                    if word in content:
                        bonus += 0.2

                return base + bonus

            ranked = sorted(chunks, key=score_chunk, reverse=True)

            # Keep only top 3 chunks
            top_chunks = ranked[:3]

            # Build compact chunk text
            compact_chunks = "\n\n".join(
                f"[CHUNK {i}] (score={c.get('score'):.4f}) Page {c['metadata'].get('page_num')}:\n"
                f"{c['content'][:1500]}..."
                for i, c in enumerate(top_chunks, 1)
            )

            # -------------------------------
            # 3) Trim Gemini analysis
            # -------------------------------
            gemini_analysis = state["gemini_analysis"]
            if len(gemini_analysis) > 2000:
                gemini_analysis = gemini_analysis[:2000] + "..."

            # -------------------------------
            # 4) Build Groq final prompt
            # -------------------------------
            prompt = self.groq_prompt.format(
                history=history_text,
                analysis=gemini_analysis,
                query=state["messages"][-1].content,
                chunks=compact_chunks
            )

            result = self.groq_llm.invoke([SystemMessage(content=prompt)])
            final_text = result.content if hasattr(result, "content") else str(result)

            final_msg = AIMessage(content=final_text.strip())

            # -------------------------------
            # 5) Save assistant reply to STM
            # -------------------------------
            self.memory_manager.store_message(doc_id, "assistant", final_msg.content)

            return {"messages": [final_msg], "sources": state["sources"]}

        except Exception as e:
            logger.error(f"[GROQ ERR] {e}")
            return {"messages": [AIMessage(content="Error generating document answer.")]}


        # ============================================================
    # SUMMARY NODE  (Writes STM + LTM)
    # ============================================================

    @log_execution_time
    def _summarizer_node(self, state: DocumentState):

        try:
            doc_id = state["doc_id"]

            # Last few exchanges
            msgs = state["messages"]
            last_msgs = [
                m for m in msgs if isinstance(m, (HumanMessage, AIMessage))
            ][-4:]

            formatted = "\n".join(
                f"{type(m).__name__}: {m.content}" for m in last_msgs
            )

            # Current summary from STM (Redis)
            current_summary = self.memory_manager.get_summary(doc_id) or ""

            # Run summary chain (Groq)
            resp = self.summary_chain.invoke({
                "current_summary": current_summary,
                "new_messages": formatted
            })

            new_summary = resp.content if hasattr(resp, "content") else str(resp)

            # -----------------------------
            # STORE TO STM + LTM
            # -----------------------------
            self.memory_manager.store_summary(
                doc_id=doc_id,
                summary=new_summary
            )

            return {"summary": new_summary}

        except Exception as e:
            logger.error(f"[SUMMARY ERR] {e}")
            return state


    # ============================================================
    # INTENT ROUTER
    # ============================================================

    def route_intent(self, intent: str, question: str, doc_id: str, user_id: str):
        """
        Direct fast-path handler for:
         - GREETING
         - DOCUMENT_SUMMARY
         - CONVERSATION_SUMMARY
         - CONVERSATION_HISTORY
         - NOT_RELATED_TO_DOCUMENT  
         - Otherwise → run RAG workflow
        """

        # -------------------------------
        # GREETING
        # -------------------------------
        if intent == "GREETING":
            return AIMessage(content="Hello! How can I help you with this document?")

        # ------------------------------------
        # NOT RELATED TO DOCUMENT
        # ------------------------------------
        if intent == "NOT_RELATED_TO_DOCUMENT":
            msg = (
                "This question is not related to the uploaded document.\n"
                "Please ask something that appears inside the document.\n"
                "For external info, use the Internet Search panel."
            )
            return AIMessage(content=msg)

        # ------------------------------------
        # DOCUMENT SUMMARY (LTM)
        # ------------------------------------
        if intent == "DOCUMENT_SUMMARY":
            summary = self.memory_manager.get_ltm_summary(doc_id)
            if not summary:
                summary = "No summary available yet. Ask questions about the document to build one."
            return AIMessage(content=summary)

        # ------------------------------------
        # CONVERSATION SUMMARY (STM)
        # ------------------------------------
        if intent == "CONVERSATION_SUMMARY":
            summary = self.memory_manager.get_summary(doc_id)
            return AIMessage(content=summary or "No recent conversation summary exists.")

        # ------------------------------------
        # LAST Q&A (LTM)
        # ------------------------------------
        if intent == "CONVERSATION_HISTORY":
            last = self.memory_manager.get_last_qa(doc_id, limit=3)
            if not last:
                return AIMessage(content="No previous document Q&A found.")
            formatted = "\n\n".join(
                f"Q: {item['question']}\nA: {item['answer']}" for item in last
            )
            return AIMessage(content=formatted)

        # ------------------------------------
        # DEFAULT → RUN FULL RAG WORKFLOW
        # ------------------------------------
        return None  # Signal workflow to proceed


    # ============================================================
    # GRAPH BUILD
    # ============================================================

    def build(self):
        try:
            g = StateGraph(DocumentState)

            g.add_node("retrieval", self._retrieval_node)
            g.add_node("gemini", self._gemini_analysis_node)
            g.add_node("groq", self._groq_response_node)
            g.add_node("summarizer", self._summarizer_node)

            g.add_edge(START, "retrieval")
            g.add_edge("retrieval", "gemini")
            g.add_edge("gemini", "groq")
            g.add_edge("groq", "summarizer")
            g.add_edge("summarizer", END)

            self.graph = g.compile()
            logger.info("[GRAPH] Document RAG workflow graph built successfully!")

        except Exception as e:
            logger.error(f"[GRAPH BUILD ERROR] {e}")
            raise WorkflowException("Graph build failed", sys)


    # ============================================================
    # GRAPH ACCESS
    # ============================================================

    def get_graph(self):
        if not self.graph:
            raise WorkflowException("Graph has not been built yet!", sys)
        return self.graph


    # ============================================================
    # CACHE
    # ============================================================

    def clear_cache(self):
        try:
            self.cache_manager.clear("gemini2.5")
            self.cache_manager.clear("groq")
        except Exception:
            pass


    # ============================================================
    # CLEANUP LIFECYCLE
    # ============================================================

    def close(self):
        """Close memory manager connections."""
        try:
            self.memory_manager.close()
        except:
            pass
