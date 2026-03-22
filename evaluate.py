"""
IR Evaluation Framework

Evaluates search engine performance using precision, recall, F1, and nDCG.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from search_engine import FinancialSearchEngine
import config


class SearchEvaluator:
    """Evaluate search engine performance with IR metrics."""
    
    def __init__(self, search_engine: FinancialSearchEngine):
        """
        Initialize evaluator.
        
        Args:
            search_engine: The search engine instance to evaluate
        """
        self.engine = search_engine
        self.test_queries = []
        self.relevance_judgments = {}
    
    def load_test_queries(self, queries_file: str):
        """
        Load test queries with relevance judgments.
        
        Expected JSON format:
        {
            "queries": [
                {
                    "id": 1,
                    "query": "How to save for retirement?",
                    "relevant_docs": [0, 1, 5],
                    "relevance_scores": [3, 2, 1]
                }
            ]
        }
        """
        with open(queries_file, 'r') as f:
            data = json.load(f)
            self.test_queries = data['queries']
    
    def precision_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Precision@K score
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not retrieved_at_k:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / len(retrieved_at_k)
    
    def recall_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Recall@K score
        """
        retrieved_at_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if not relevant_set:
            return 0.0
        
        return len(retrieved_at_k & relevant_set) / len(relevant_set)
    
    def f1_at_k(self, retrieved: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate F1@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def dcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain@K.
        
        Args:
            relevance_scores: Relevance scores in retrieved order
            k: Cutoff rank
            
        Returns:
            DCG@K score
        """
        relevance_scores = relevance_scores[:k]
        if not relevance_scores:
            return 0.0
        
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, relevance_scores: List[float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            relevance_scores: Relevance scores in retrieved order
            k: Cutoff rank
            
        Returns:
            nDCG@K score (0-1, higher is better)
        """
        dcg = self.dcg_at_k(relevance_scores, k)
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = self.dcg_at_k(ideal_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, query: str, relevant_docs: List[int], 
                      relevance_scores: List[float], k: int = 5) -> Dict:
        """
        Evaluate a single query.
        
        Args:
            query: The search query
            relevant_docs: IDs of relevant documents
            relevance_scores: Relevance score for each relevant doc (e.g., 3=very, 2=somewhat, 1=marginally)
            k: Number of results to retrieve
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = self.engine.search(query, k=k)
        retrieved_ids = list(range(len(results)))
        
        retrieved_relevance = []
        for doc_id in retrieved_ids:
            if doc_id in relevant_docs:
                idx = relevant_docs.index(doc_id)
                retrieved_relevance.append(relevance_scores[idx])
            else:
                retrieved_relevance.append(0.0)
        
        metrics = {
            "query": query,
            "num_results": len(results),
            "precision@5": self.precision_at_k(retrieved_ids, relevant_docs, 5),
            "recall@5": self.recall_at_k(retrieved_ids, relevant_docs, 5),
            "f1@5": self.f1_at_k(retrieved_ids, relevant_docs, 5),
            "precision@10": self.precision_at_k(retrieved_ids, relevant_docs, 10),
            "recall@10": self.recall_at_k(retrieved_ids, relevant_docs, 10),
            "f1@10": self.f1_at_k(retrieved_ids, relevant_docs, 10),
            "ndcg@5": self.ndcg_at_k(retrieved_relevance, 5),
            "ndcg@10": self.ndcg_at_k(retrieved_relevance, 10),
        }
        
        return metrics
    
    def run_evaluation(self) -> Dict:
        """
        Run evaluation on all test queries.
        
        Returns:
            Dictionary with per-query and aggregate metrics
        """
        if not self.test_queries:
            print("No test queries loaded")
            return {}
        
        results = []
        
        print("\n" + "="*60)
        print("Running Evaluation")
        print("="*60 + "\n")
        
        for test_case in self.test_queries:
            query = test_case['query']
            relevant = test_case['relevant_docs']
            scores = test_case.get('relevance_scores', [1] * len(relevant))
            
            metrics = self.evaluate_query(query, relevant, scores, k=5)
            results.append(metrics)
            
            print(f"Query: {query}")
            print(f"  P@5: {metrics['precision@5']:.3f}  R@5: {metrics['recall@5']:.3f}  F1@5: {metrics['f1@5']:.3f}")
            print(f"  P@10: {metrics['precision@10']:.3f}  R@10: {metrics['recall@10']:.3f}  F1@10: {metrics['f1@10']:.3f}")
            print(f"  nDCG@5: {metrics['ndcg@5']:.3f}  nDCG@10: {metrics['ndcg@10']:.3f}")
            print()
        
        avg_metrics = {
            "avg_precision@5": np.mean([r['precision@5'] for r in results]),
            "avg_recall@5": np.mean([r['recall@5'] for r in results]),
            "avg_f1@5": np.mean([r['f1@5'] for r in results]),
            "avg_precision@10": np.mean([r['precision@10'] for r in results]),
            "avg_recall@10": np.mean([r['recall@10'] for r in results]),
            "avg_f1@10": np.mean([r['f1@10'] for r in results]),
            "avg_ndcg@5": np.mean([r['ndcg@5'] for r in results]),
            "avg_ndcg@10": np.mean([r['ndcg@10'] for r in results]),
        }
        
        print("="*60)
        print("Average Metrics Across All Queries")
        print("="*60)
        print(f"  Avg P@5: {avg_metrics['avg_precision@5']:.3f}")
        print(f"  Avg R@5: {avg_metrics['avg_recall@5']:.3f}")
        print(f"  Avg F1@5: {avg_metrics['avg_f1@5']:.3f}")
        print(f"  Avg nDCG@5: {avg_metrics['avg_ndcg@5']:.3f}")
        print()
        print(f"  Avg P@10: {avg_metrics['avg_precision@10']:.3f}")
        print(f"  Avg R@10: {avg_metrics['avg_recall@10']:.3f}")
        print(f"  Avg F1@10: {avg_metrics['avg_f1@10']:.3f}")
        print(f"  Avg nDCG@10: {avg_metrics['avg_ndcg@10']:.3f}")
        
        return {
            "per_query": results,
            "aggregate": avg_metrics
        }


if __name__ == "__main__":
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY == "your-api-key-here":
        print("❌ Error: OPENAI_API_KEY not set")
        print("\nEdit config.py and set your API key, or:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)
    
    engine = FinancialSearchEngine(openai_api_key=config.OPENAI_API_KEY)
    
    if engine.load_index():
        evaluator = SearchEvaluator(engine)
        evaluator.load_test_queries("test_queries.json")
        evaluator.run_evaluation()
    else:
        print("No index found. Run search_engine.py first to build the index.")
