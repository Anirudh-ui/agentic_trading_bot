"""
CAG (Corrective-RAG) Evaluator
Implements relevancy metrics including MMR, semantic similarity, and relevance scoring
"""

from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

class CAGEvaluator:
    """
    Evaluates retrieved documents for relevance and implements corrective actions
    """
    
    def __init__(self, relevance_threshold: float = 0.5):
        self.relevance_threshold = relevance_threshold
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def calculate_relevance_score(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate relevance scores for documents using TF-IDF and cosine similarity
        
        Args:
            query: User query
            documents: List of retrieved documents
            
        Returns:
            List of relevance scores (0-1) for each document
        """
        if not documents:
            return []
        
        try:
            # Combine query with documents for vectorization
            all_texts = [query] + documents
            
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Query vector is the first one
            query_vector = tfidf_matrix[0:1]
            
            # Document vectors are the rest
            doc_vectors = tfidf_matrix[1:]
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            return similarities.tolist()
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return [0.0] * len(documents)
    
    def calculate_mmr_scores(
        self, 
        query: str, 
        documents: List[str], 
        lambda_param: float = 0.7
    ) -> List[float]:
        """
        Calculate Maximum Marginal Relevance (MMR) scores
        Balances relevance with diversity
        
        Args:
            query: User query
            documents: List of retrieved documents
            lambda_param: Balance parameter (0=diversity, 1=relevance)
            
        Returns:
            List of MMR scores for each document
        """
        if not documents:
            return []
        
        try:
            # Get relevance scores
            relevance_scores = self.calculate_relevance_score(query, documents)
            
            # Calculate document-document similarities for diversity
            all_docs = [query] + documents
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            doc_vectors = tfidf_matrix[1:]
            
            # Pairwise document similarities
            doc_similarities = cosine_similarity(doc_vectors)
            
            # Calculate MMR scores
            mmr_scores = []
            selected_indices = []
            
            for i in range(len(documents)):
                if not selected_indices:
                    # First document - just use relevance
                    mmr_scores.append(relevance_scores[i])
                else:
                    # Balance relevance and diversity
                    max_sim_to_selected = max(
                        doc_similarities[i][j] for j in selected_indices
                    )
                    mmr = (
                        lambda_param * relevance_scores[i] - 
                        (1 - lambda_param) * max_sim_to_selected
                    )
                    mmr_scores.append(mmr)
                
                selected_indices.append(i)
            
            return mmr_scores
        except Exception as e:
            print(f"Error calculating MMR: {e}")
            return [0.0] * len(documents)
    
    def keyword_overlap_score(self, query: str, document: str) -> float:
        """
        Calculate keyword overlap between query and document
        Simple but fast relevance metric
        """
        query_words = set(self._tokenize(query.lower()))
        doc_words = set(self._tokenize(document.lower()))
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(doc_words)
        
        # Jaccard similarity
        union = query_words.union(doc_words)
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def semantic_similarity(self, query: str, document: str) -> float:
        """
        Calculate semantic similarity using TF-IDF
        """
        try:
            texts = [query, document]
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def evaluate_retrieval(
        self, 
        query: str, 
        documents: List[str], 
        use_mmr: bool = True
    ) -> dict:
        """
        Comprehensive evaluation of retrieved documents
        
        Returns:
            Dictionary with scores and recommendations
        """
        if not documents:
            return {
                'needs_correction': True,
                'reason': 'No documents retrieved',
                'avg_relevance': 0.0,
                'recommendation': 'use_web_search'
            }
        
        # Calculate scores
        if use_mmr:
            scores = self.calculate_mmr_scores(query, documents)
        else:
            scores = self.calculate_relevance_score(query, documents)
        
        avg_score = np.mean(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        # Determine if correction is needed
        needs_correction = avg_score < self.relevance_threshold
        
        # Provide recommendation
        if needs_correction:
            if max_score < 0.3:
                recommendation = 'use_web_search'
                reason = 'Very low relevance - knowledge base has no relevant info'
            else:
                recommendation = 'refine_query'
                reason = 'Moderate relevance - try rephrasing query'
        else:
            recommendation = 'use_retrieved'
            reason = 'Good relevance - use retrieved documents'
        
        return {
            'needs_correction': needs_correction,
            'reason': reason,
            'avg_relevance': float(avg_score),
            'max_relevance': float(max_score),
            'min_relevance': float(min(scores)) if scores else 0.0,
            'scores': scores,
            'recommendation': recommendation,
            'num_documents': len(documents)
        }
    
    def rank_documents(
        self, 
        query: str, 
        documents: List[str],
        use_mmr: bool = True
    ) -> List[Tuple[int, str, float]]:
        """
        Rank documents by relevance/MMR score
        
        Returns:
            List of (index, document, score) tuples, sorted by score
        """
        if use_mmr:
            scores = self.calculate_mmr_scores(query, documents)
        else:
            scores = self.calculate_relevance_score(query, documents)
        
        # Combine documents with scores and indices
        ranked = list(zip(range(len(documents)), documents, scores))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[2], reverse=True)
        
        return ranked
    
    def filter_by_threshold(
        self, 
        query: str, 
        documents: List[str],
        threshold: float = None
    ) -> List[str]:
        """
        Filter documents that meet relevance threshold
        """
        if threshold is None:
            threshold = self.relevance_threshold
        
        scores = self.calculate_relevance_score(query, documents)
        
        filtered = [
            doc for doc, score in zip(documents, scores)
            if score >= threshold
        ]
        
        return filtered


class RelevancyMetrics:
    """
    Track and report relevancy metrics over time
    """
    
    def __init__(self):
        self.queries = []
        self.relevance_scores = []
        self.corrections_made = 0
        self.total_queries = 0
    
    def record_query(self, query: str, avg_relevance: float, needs_correction: bool):
        """Record a query and its relevance metrics"""
        self.queries.append(query)
        self.relevance_scores.append(avg_relevance)
        self.total_queries += 1
        
        if needs_correction:
            self.corrections_made += 1
    
    def get_statistics(self) -> dict:
        """Get aggregated statistics"""
        if not self.relevance_scores:
            return {
                'total_queries': 0,
                'avg_relevance': 0.0,
                'correction_rate': 0.0
            }
        
        return {
            'total_queries': self.total_queries,
            'avg_relevance': np.mean(self.relevance_scores),
            'median_relevance': np.median(self.relevance_scores),
            'min_relevance': min(self.relevance_scores),
            'max_relevance': max(self.relevance_scores),
            'correction_rate': self.corrections_made / self.total_queries * 100,
            'corrections_made': self.corrections_made
        }
    
    def reset(self):
        """Reset all metrics"""
        self.queries = []
        self.relevance_scores = []
        self.corrections_made = 0
        self.total_queries = 0


# Global metrics tracker
GLOBAL_METRICS = RelevancyMetrics()


if __name__ == "__main__":
    # Test the evaluator
    evaluator = CAGEvaluator(relevance_threshold=0.5)
    
    query = "What is a bull market?"
    documents = [
        "A bull market is a financial market condition where prices are rising or expected to rise.",
        "The weather today is sunny with a chance of rain.",
        "Bull markets are characterized by investor optimism and confidence."
    ]
    
    print("=" * 70)
    print("CAG EVALUATOR TEST")
    print("=" * 70)
    
    # Test relevance scoring
    print(f"\nQuery: {query}\n")
    
    result = evaluator.evaluate_retrieval(query, documents, use_mmr=True)
    
    print(f"Average Relevance: {result['avg_relevance']:.3f}")
    print(f"Max Relevance: {result['max_relevance']:.3f}")
    print(f"Needs Correction: {result['needs_correction']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Reason: {result['reason']}")
    
    # Test ranking
    print("\n" + "=" * 70)
    print("DOCUMENT RANKING")
    print("=" * 70)
    
    ranked = evaluator.rank_documents(query, documents)
    for i, (idx, doc, score) in enumerate(ranked, 1):
        print(f"\nRank {i} (Score: {score:.3f}):")
        print(f"  {doc[:100]}...")