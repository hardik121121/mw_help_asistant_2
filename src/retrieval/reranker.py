"""
Cohere Reranker for precision improvement.
"""

import logging
import time
from typing import List, Dict, Optional

try:
    import cohere
except ImportError:
    print("⚠️  Cohere library not installed. Please run: pip install cohere")
    cohere = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class CohereReranker:
    """
    Cohere-based reranking for search result precision.

    Features:
    - Semantic reranking using Cohere API
    - Configurable top-k
    - Diversity enforcement
    - Rate limiting
    - Fallback to original ranking
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize Cohere reranker.

        Args:
            model: Cohere rerank model (default from settings)
        """
        self.settings = get_settings()

        if cohere is None:
            raise RuntimeError("Cohere library not installed")

        self.client = cohere.Client(api_key=self.settings.cohere_api_key)
        self.model = model or self.settings.rerank_model

        logger.info(f"Initialized CohereReranker with model: {self.model}")

    def rerank(self,
               query: str,
               documents: List[Dict],
               top_k: Optional[int] = None,
               enforce_diversity: bool = False,
               diversity_threshold: float = 0.7) -> List[Dict]:
        """
        Rerank documents using Cohere.

        Args:
            query: Query text
            documents: List of document dicts with 'content' field
            top_k: Number of results to return (default from settings)
            enforce_diversity: Whether to enforce diversity in results
            diversity_threshold: Similarity threshold for diversity (0-1)

        Returns:
            Reranked list of documents with new scores
        """
        top_k = top_k or self.settings.rerank_top_k

        if not documents:
            logger.warning("No documents to rerank")
            return []

        logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        try:
            # Prepare documents for Cohere API
            texts = [doc['content'] for doc in documents]

            # Call Cohere rerank API
            start_time = time.time()
            response = self.client.rerank(
                model=self.model,
                query=query,
                documents=texts,
                top_n=min(top_k * 2, len(documents)) if enforce_diversity else top_k,
                return_documents=False  # We already have the documents
            )
            elapsed = time.time() - start_time

            logger.info(f"  Cohere rerank completed in {elapsed:.2f}s")

            # Build reranked results
            reranked_results = []
            for result in response.results:
                doc = documents[result.index]
                doc['rerank_score'] = float(result.relevance_score)
                doc['original_rank'] = result.index
                doc['score'] = float(result.relevance_score)  # Update score
                reranked_results.append(doc)

            # Apply diversity if requested
            if enforce_diversity and len(reranked_results) > top_k:
                reranked_results = self._enforce_diversity(
                    reranked_results,
                    top_k=top_k,
                    threshold=diversity_threshold
                )

            # Return final top-k
            final_results = reranked_results[:top_k]

            logger.info(f"✅ Reranking complete: {len(final_results)} results")
            logger.info(f"  Top score: {final_results[0]['rerank_score']:.4f}")
            logger.info(f"  Lowest score: {final_results[-1]['rerank_score']:.4f}")

            # Rate limiting
            time.sleep(self.settings.cohere_rate_limit)

            return final_results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Falling back to original ranking")
            return documents[:top_k]

    def _enforce_diversity(self,
                          documents: List[Dict],
                          top_k: int,
                          threshold: float = 0.7) -> List[Dict]:
        """
        Enforce diversity by removing very similar documents.

        Uses heading path and page proximity as diversity signals.

        Args:
            documents: Ranked documents
            top_k: Target number of results
            threshold: Similarity threshold (0-1)

        Returns:
            Diverse subset of documents
        """
        logger.info(f"  Enforcing diversity (threshold: {threshold})")

        diverse_docs = []
        seen_sections = set()
        seen_pages = set()

        for doc in documents:
            if len(diverse_docs) >= top_k:
                break

            metadata = doc.get('metadata', {})

            # Get diversity signals
            heading_path = tuple(metadata.get('heading_path', []))
            page_start = metadata.get('page_start', -1)

            # Check section diversity
            section_key = heading_path[:2] if len(heading_path) >= 2 else heading_path
            is_duplicate_section = section_key in seen_sections and len(seen_sections) > 0

            # Check page proximity
            is_nearby_page = any(
                abs(page_start - seen_page) <= self.settings.page_proximity
                for seen_page in seen_pages
            ) if page_start >= 0 else False

            # Add if diverse enough
            if not (is_duplicate_section and is_nearby_page):
                diverse_docs.append(doc)
                seen_sections.add(section_key)
                if page_start >= 0:
                    seen_pages.add(page_start)
            elif len(diverse_docs) < top_k // 2:
                # Always include at least half the results
                diverse_docs.append(doc)

        logger.info(f"  Diversity: {len(documents)} -> {len(diverse_docs)} docs")

        # Fill up to top_k if needed
        if len(diverse_docs) < top_k:
            remaining = [d for d in documents if d not in diverse_docs]
            diverse_docs.extend(remaining[:top_k - len(diverse_docs)])

        return diverse_docs

    def batch_rerank(self,
                     queries: List[str],
                     documents_list: List[List[Dict]],
                     top_k: Optional[int] = None) -> List[List[Dict]]:
        """
        Rerank multiple query-document sets.

        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of results per query

        Returns:
            List of reranked document lists
        """
        logger.info(f"Batch reranking {len(queries)} queries")

        results = []
        for i, (query, documents) in enumerate(zip(queries, documents_list), 1):
            logger.info(f"  Reranking query {i}/{len(queries)}")
            reranked = self.rerank(query, documents, top_k=top_k)
            results.append(reranked)

        logger.info("✅ Batch reranking complete")
        return results


if __name__ == "__main__":
    """Test Cohere reranker."""
    print("\n" + "="*70)
    print("TESTING COHERE RERANKER")
    print("="*70 + "\n")

    # Initialize
    reranker = CohereReranker()

    # Test query
    test_query = "How do I create a no-code block on Watermelon?"
    print(f"Test Query: {test_query}\n")

    # Get some documents using hybrid search
    from src.retrieval.hybrid_search import HybridSearch
    from src.database.embedding_generator import EmbeddingGenerator

    hybrid_search = HybridSearch()
    generator = EmbeddingGenerator()

    query_embedding = generator.generate_embeddings([test_query])[0]
    documents = hybrid_search.search(
        query=test_query,
        query_embedding=query_embedding,
        top_k=30
    )

    print(f"Initial results: {len(documents)} documents\n")

    # Rerank
    reranked = reranker.rerank(
        query=test_query,
        documents=documents,
        top_k=10,
        enforce_diversity=True
    )

    # Display results
    print(f"\n📊 Reranked Results ({len(reranked)} chunks):\n")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Chunk: {result['chunk_id']}")
        print(f"   Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Original Rank: {result.get('original_rank', 'N/A')}")
        print(f"   Heading: {' > '.join(result['metadata'].get('heading_path', []))}")
        print(f"   Content: {result['content'][:100]}...")
        print()

    print("✅ Reranker test complete!")
