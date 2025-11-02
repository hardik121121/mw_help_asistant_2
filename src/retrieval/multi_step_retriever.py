"""
Multi-Step Retriever for complex query handling.
Coordinates retrieval for decomposed queries with context chaining.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import CohereReranker
from src.retrieval.context_organizer import ContextOrganizer, OrganizedContext
from src.database.embedding_generator import EmbeddingGenerator
from src.query.query_understanding import QueryUnderstanding
from src.query.query_decomposer import SubQuestion

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Complete retrieval result for a query.

    Attributes:
        query: Original query
        query_understanding: Query analysis results
        organized_context: Final organized context
        sub_results: Results per sub-question
        total_chunks_retrieved: Total chunks before deduplication
        final_chunks: Number of chunks in final context
        retrieval_time: Time taken (seconds)
    """
    query: str
    query_understanding: QueryUnderstanding
    organized_context: OrganizedContext
    sub_results: List[Dict] = field(default_factory=list)
    total_chunks_retrieved: int = 0
    final_chunks: int = 0
    retrieval_time: float = 0.0


class MultiStepRetriever:
    """
    Multi-step retrieval system for complex queries.

    Features:
    - Per-sub-question retrieval
    - Context chaining between steps
    - Hybrid search (vector + BM25)
    - Cohere reranking
    - Result aggregation and organization
    """

    def __init__(self,
                 use_reranking: bool = True,
                 enable_context_chaining: bool = True):
        """
        Initialize multi-step retriever.

        Args:
            use_reranking: Whether to use Cohere reranking
            enable_context_chaining: Use results from previous steps
        """
        logger.info("Initializing MultiStepRetriever")

        self.hybrid_search = HybridSearch()
        self.embedding_generator = EmbeddingGenerator()
        self.context_organizer = ContextOrganizer()

        self.use_reranking = use_reranking
        self.enable_context_chaining = enable_context_chaining

        if use_reranking:
            self.reranker = CohereReranker()
            logger.info("  Reranking: ENABLED")
        else:
            self.reranker = None
            logger.info("  Reranking: DISABLED")

        logger.info(f"  Context chaining: {'ENABLED' if enable_context_chaining else 'DISABLED'}")
        logger.info("✅ MultiStepRetriever initialized")

    def retrieve(self,
                query: str,
                query_understanding: QueryUnderstanding,
                max_chunks: int = 20) -> RetrievalResult:
        """
        Perform multi-step retrieval for a complex query.

        Args:
            query: Original query text
            query_understanding: Query analysis from Phase 3
            max_chunks: Maximum chunks in final context

        Returns:
            RetrievalResult with organized context
        """
        import time
        start_time = time.time()

        logger.info("="*70)
        logger.info(f"MULTI-STEP RETRIEVAL: {query}")
        logger.info("="*70)

        # Check if query needs decomposition
        sub_questions = query_understanding.decomposition.sub_questions

        if not sub_questions or len(sub_questions) == 0:
            logger.info("Simple query - using single-step retrieval")
            from src.query.query_decomposer import DependencyType
            sub_questions = [
                SubQuestion(
                    id="q1",
                    question=query,
                    topics=["general"],
                    dependency_type=DependencyType.INDEPENDENT,
                    priority=1
                )
            ]

        logger.info(f"\n📋 Sub-questions: {len(sub_questions)}")
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {sq.question}")
            logger.info(f"     Topics: {', '.join(sq.topics)}, Dependency: {sq.dependency_type}")

        # Retrieve for each sub-question
        results_by_subquestion = []
        context_chain = []  # For context chaining

        for i, sub_question in enumerate(sub_questions, 1):
            logger.info(f"\n🔍 Retrieving for sub-question {i}/{len(sub_questions)}")
            logger.info(f"   {sub_question.question}")

            # Enhance query with context from previous steps if enabled
            enhanced_query = self._enhance_with_context(
                sub_question.question,
                context_chain
            ) if self.enable_context_chaining and context_chain else sub_question.question

            # Retrieve
            sub_results = self._retrieve_for_subquestion(
                enhanced_query,
                sub_question
            )

            results_by_subquestion.append(sub_results)
            logger.info(f"   Retrieved: {len(sub_results)} chunks")

            # Add top results to context chain for next iteration
            if self.enable_context_chaining and sub_results:
                context_chain.extend(sub_results[:3])  # Top 3 for context

        # Organize results
        logger.info("\n📊 Organizing context...")
        organized_context = self.context_organizer.organize(
            results_by_subquestion,
            max_chunks=max_chunks
        )

        # Build result
        elapsed = time.time() - start_time
        total_retrieved = sum(len(results) for results in results_by_subquestion)

        result = RetrievalResult(
            query=query,
            query_understanding=query_understanding,
            organized_context=organized_context,
            sub_results=[
                {
                    'sub_question': sq.question,
                    'num_results': len(results)
                }
                for sq, results in zip(sub_questions, results_by_subquestion)
            ],
            total_chunks_retrieved=total_retrieved,
            final_chunks=organized_context.total_chunks,
            retrieval_time=elapsed
        )

        logger.info("\n" + "="*70)
        logger.info("✅ MULTI-STEP RETRIEVAL COMPLETE")
        logger.info("="*70)
        logger.info(f"Total retrieved: {total_retrieved} chunks")
        logger.info(f"Final context: {organized_context.total_chunks} chunks")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info("="*70 + "\n")

        return result

    def _retrieve_for_subquestion(self,
                                  query: str,
                                  sub_question: SubQuestion) -> List[Dict]:
        """
        Retrieve relevant chunks for a single sub-question.

        Args:
            query: Query text (possibly enhanced with context)
            sub_question: SubQuestion object

        Returns:
            List of retrieved chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        # Hybrid search
        results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=30,  # Get more for reranking
            filter_toc=True
        )

        logger.info(f"     Hybrid search: {len(results)} results")

        # Rerank if enabled
        if self.use_reranking and self.reranker and results:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=10,
                enforce_diversity=True
            )
            logger.info(f"     After reranking: {len(results)} results")

        return results

    def _enhance_with_context(self,
                             query: str,
                             context_chain: List[Dict]) -> str:
        """
        Enhance query with context from previous retrieval steps.

        Args:
            query: Original query
            context_chain: Previous retrieval results

        Returns:
            Enhanced query string
        """
        if not context_chain:
            return query

        # Extract key terms from previous results
        context_terms = set()
        for chunk in context_chain[:3]:  # Top 3 chunks
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])

            # Add heading terms
            for heading in heading_path:
                # Simple term extraction (could be enhanced)
                terms = heading.lower().split()
                context_terms.update([t for t in terms if len(t) > 4])

        # Enhance query with top terms
        if context_terms:
            top_terms = list(context_terms)[:5]
            enhanced = f"{query} (related to: {', '.join(top_terms)})"
            logger.info(f"     Enhanced query with context: {top_terms[:3]}")
            return enhanced

        return query

    def retrieve_simple(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Simple single-step retrieval (for non-complex queries).

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of retrieved chunks
        """
        logger.info(f"Simple retrieval for: {query}")

        # Generate embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]

        # Hybrid search
        results = self.hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k * 3
        )

        # Rerank if enabled
        if self.use_reranking and self.reranker and results:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=top_k
            )

        logger.info(f"✅ Retrieved {len(results)} chunks")
        return results


if __name__ == "__main__":
    """Test multi-step retriever."""
    print("\n" + "="*70)
    print("TESTING MULTI-STEP RETRIEVER")
    print("="*70 + "\n")

    # Initialize retriever
    retriever = MultiStepRetriever(
        use_reranking=True,
        enable_context_chaining=True
    )

    # Test with query understanding
    from src.query.query_understanding import QueryUnderstandingEngine

    # Test query
    test_query = "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"
    print(f"Test Query: {test_query}\n")

    # Understand query
    print("Understanding query...")
    query_system = QueryUnderstandingEngine()
    understanding = query_system.understand(test_query)

    print(f"\nQuery Analysis:")
    print(f"  Type: {understanding.classification.query_type}")
    print(f"  Complexity: {understanding.classification.complexity}")
    print(f"  Sub-questions: {len(understanding.decomposition.sub_questions)}")
    for i, sq in enumerate(understanding.decomposition.sub_questions, 1):
        print(f"    {i}. {sq.question}")

    # Retrieve
    print("\n" + "="*70)
    print("Starting retrieval...\n")

    result = retriever.retrieve(
        query=test_query,
        query_understanding=understanding,
        max_chunks=15
    )

    # Display results
    print(f"\n📊 Retrieval Results:\n")
    print(f"Total Retrieved: {result.total_chunks_retrieved} chunks")
    print(f"Final Context: {result.final_chunks} chunks")
    print(f"Time: {result.retrieval_time:.2f}s")
    print(f"\nSub-question results:")
    for sr in result.sub_results:
        print(f"  - {sr['sub_question'][:50]}... ({sr['num_results']} chunks)")

    print(f"\nOrganized Context:")
    print(f"  Unique Sections: {result.organized_context.unique_sections}")
    print(f"  Page Range: {result.organized_context.page_range}")
    print(f"  Has Images: {result.organized_context.has_images}")
    print(f"  Has Tables: {result.organized_context.has_tables}")

    print(f"\n  Topic Distribution:")
    for topic, chunks in result.organized_context.topic_groups.items():
        print(f"    - {topic}: {len(chunks)} chunks")

    print("\n" + "="*70)
    print("✅ Multi-step retriever test complete!")
    print("="*70 + "\n")
