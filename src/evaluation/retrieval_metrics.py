"""
Retrieval Metrics Calculator.
Evaluates retrieval quality using standard IR metrics.
"""

import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import math

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """
    Retrieval evaluation metrics.

    Attributes:
        precision_at_k: Precision at various k values
        recall_at_k: Recall at various k values
        mrr: Mean Reciprocal Rank
        map_score: Mean Average Precision
        ndcg_at_k: Normalized Discounted Cumulative Gain at k
        coverage: Topic coverage score
        diversity: Result diversity score
        total_retrieved: Total chunks retrieved
        relevant_retrieved: Number of relevant chunks
    """
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    coverage: float = 0.0
    diversity: float = 0.0
    total_retrieved: int = 0
    relevant_retrieved: int = 0


class RetrievalMetricsCalculator:
    """
    Calculate standard information retrieval metrics.

    Metrics:
    - Precision@K: Proportion of relevant docs in top-K
    - Recall@K: Proportion of relevant docs found in top-K
    - MRR: Mean Reciprocal Rank
    - MAP: Mean Average Precision
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Coverage: Topic coverage
    - Diversity: Result diversity
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize metrics calculator.

        Args:
            k_values: List of k values for metrics (default: [5, 10, 20])
        """
        self.k_values = k_values or [5, 10, 20]
        logger.info(f"Initialized RetrievalMetricsCalculator with k={self.k_values}")

    def calculate_metrics(self,
                         retrieved_chunks: List[Dict],
                         expected_topics: List[str],
                         query_data: Dict = None) -> RetrievalMetrics:
        """
        Calculate all retrieval metrics.

        Args:
            retrieved_chunks: List of retrieved chunk dicts
            expected_topics: List of expected topic keywords
            query_data: Optional query metadata for ground truth

        Returns:
            RetrievalMetrics with all scores
        """
        logger.info("Calculating retrieval metrics...")

        # Determine relevance (simplified - based on topic matching)
        relevant_chunks = self._identify_relevant_chunks(
            retrieved_chunks, expected_topics
        )

        metrics = RetrievalMetrics()
        metrics.total_retrieved = len(retrieved_chunks)
        metrics.relevant_retrieved = len(relevant_chunks)

        # Calculate precision and recall at K
        for k in self.k_values:
            precision = self._precision_at_k(retrieved_chunks, relevant_chunks, k)
            recall = self._recall_at_k(retrieved_chunks, relevant_chunks, k)
            ndcg = self._ndcg_at_k(retrieved_chunks, relevant_chunks, k)

            metrics.precision_at_k[k] = precision
            metrics.recall_at_k[k] = recall
            metrics.ndcg_at_k[k] = ndcg

        # Calculate MRR
        metrics.mrr = self._mean_reciprocal_rank(retrieved_chunks, relevant_chunks)

        # Calculate MAP
        metrics.map_score = self._mean_average_precision(retrieved_chunks, relevant_chunks)

        # Calculate coverage
        metrics.coverage = self._calculate_coverage(retrieved_chunks, expected_topics)

        # Calculate diversity
        metrics.diversity = self._calculate_diversity(retrieved_chunks)

        logger.info(f"  Precision@10: {metrics.precision_at_k.get(10, 0):.3f}")
        logger.info(f"  Recall@10: {metrics.recall_at_k.get(10, 0):.3f}")
        logger.info(f"  MRR: {metrics.mrr:.3f}")
        logger.info(f"  Coverage: {metrics.coverage:.3f}")

        return metrics

    def _identify_relevant_chunks(self,
                                  chunks: List[Dict],
                                  expected_topics: List[str]) -> Set[str]:
        """
        Identify relevant chunks based on topic matching.

        This is a simplified relevance judgment based on keyword matching.
        In a production system, you'd want human-labeled ground truth.

        Args:
            chunks: Retrieved chunks
            expected_topics: Expected topic keywords

        Returns:
            Set of relevant chunk IDs
        """
        relevant_ids = set()

        for chunk in chunks:
            content = chunk.get('content', '').lower()
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])

            # Check if chunk is relevant based on topics
            chunk_text = content + ' ' + ' '.join(heading_path).lower()

            for topic in expected_topics:
                topic_keywords = topic.lower().split()
                # If any keyword from topic appears in chunk
                if any(kw in chunk_text for kw in topic_keywords):
                    relevant_ids.add(chunk['chunk_id'])
                    break

        return relevant_ids

    def _precision_at_k(self,
                       retrieved: List[Dict],
                       relevant: Set[str],
                       k: int) -> float:
        """Calculate Precision@K."""
        if k == 0 or not retrieved:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(
            1 for chunk in top_k
            if chunk['chunk_id'] in relevant
        )

        return relevant_in_top_k / k

    def _recall_at_k(self,
                    retrieved: List[Dict],
                    relevant: Set[str],
                    k: int) -> float:
        """Calculate Recall@K."""
        if not relevant:
            return 0.0

        top_k = retrieved[:k]
        relevant_in_top_k = sum(
            1 for chunk in top_k
            if chunk['chunk_id'] in relevant
        )

        return relevant_in_top_k / len(relevant)

    def _mean_reciprocal_rank(self,
                             retrieved: List[Dict],
                             relevant: Set[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, chunk in enumerate(retrieved, 1):
            if chunk['chunk_id'] in relevant:
                return 1.0 / i

        return 0.0

    def _mean_average_precision(self,
                               retrieved: List[Dict],
                               relevant: Set[str]) -> float:
        """Calculate Mean Average Precision."""
        if not relevant:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, chunk in enumerate(retrieved, 1):
            if chunk['chunk_id'] in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        if relevant_count == 0:
            return 0.0

        return precision_sum / len(relevant)

    def _ndcg_at_k(self,
                  retrieved: List[Dict],
                  relevant: Set[str],
                  k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at K."""
        if k == 0 or not retrieved:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, chunk in enumerate(retrieved[:k], 1):
            relevance = 1.0 if chunk['chunk_id'] in relevant else 0.0
            dcg += relevance / math.log2(i + 1)

        # Calculate Ideal DCG
        ideal_relevance = [1.0] * min(len(relevant), k)
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_coverage(self,
                           chunks: List[Dict],
                           expected_topics: List[str]) -> float:
        """
        Calculate topic coverage.

        What proportion of expected topics are covered in results?
        """
        if not expected_topics:
            return 1.0

        covered_topics = set()

        for chunk in chunks:
            content = chunk.get('content', '').lower()
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])
            chunk_text = content + ' ' + ' '.join(heading_path).lower()

            for topic in expected_topics:
                topic_keywords = topic.lower().split()
                if any(kw in chunk_text for kw in topic_keywords):
                    covered_topics.add(topic)

        return len(covered_topics) / len(expected_topics)

    def _calculate_diversity(self, chunks: List[Dict]) -> float:
        """
        Calculate result diversity.

        Measures how many unique sections are represented.
        """
        if not chunks:
            return 0.0

        unique_sections = set()

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])

            if heading_path:
                # Use top-level section
                section = heading_path[0]
                unique_sections.add(section)

        # Normalize by a reasonable expectation
        # More than 5 unique sections is excellent diversity
        return min(len(unique_sections) / 5.0, 1.0)


if __name__ == "__main__":
    """Test retrieval metrics calculator."""
    print("\n" + "="*70)
    print("TESTING RETRIEVAL METRICS CALCULATOR")
    print("="*70 + "\n")

    calculator = RetrievalMetricsCalculator()
    print(f"✅ RetrievalMetricsCalculator initialized")
    print(f"   K values: {calculator.k_values}")

    # Simulate some retrieved chunks
    test_chunks = [
        {
            'chunk_id': 'chunk_1',
            'content': 'Information about no-code blocks',
            'metadata': {'heading_path': ['Getting Started', 'No-Code Blocks']}
        },
        {
            'chunk_id': 'chunk_2',
            'content': 'Details on testing framework',
            'metadata': {'heading_path': ['Advanced', 'Testing']}
        },
    ]

    test_topics = ['no-code blocks', 'testing']

    metrics = calculator.calculate_metrics(test_chunks, test_topics)

    print(f"\n📊 Sample Metrics:")
    print(f"  Precision@5: {metrics.precision_at_k.get(5, 0):.3f}")
    print(f"  Recall@5: {metrics.recall_at_k.get(5, 0):.3f}")
    print(f"  MRR: {metrics.mrr:.3f}")
    print(f"  Coverage: {metrics.coverage:.3f}")
    print(f"  Diversity: {metrics.diversity:.3f}")

    print("\n✅ Retrieval metrics calculator ready!\n")
