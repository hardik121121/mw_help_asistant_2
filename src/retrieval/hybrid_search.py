"""
Hybrid Search Engine combining Vector and BM25 search with RRF fusion.
"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

from config.settings import get_settings
from src.query.query_expander import QueryExpander

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Hybrid search combining vector and keyword search.

    Features:
    - Vector search via Pinecone
    - BM25 keyword search
    - Reciprocal Rank Fusion (RRF) for result merging
    - Metadata filtering
    - TOC chunk filtering
    """

    def __init__(self):
        """Initialize hybrid search."""
        self.settings = get_settings()

        # Load Pinecone index
        if Pinecone is None:
            raise RuntimeError("Pinecone library not installed")

        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = self.pc.Index(self.settings.pinecone_index_name)

        # Load BM25 index
        bm25_path = self.settings.cache_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at: {bm25_path}")

        logger.info(f"Loading BM25 index from: {bm25_path}")
        with open(bm25_path, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
            self.bm25_chunks = bm25_data['chunks']
            self.tokenized_corpus = bm25_data['tokenized_corpus']

        logger.info(f"Loaded BM25 index with {len(self.bm25_chunks)} chunks")

        # Load embeddings for vector search (optional, mainly needed for query embeddings)
        # Embeddings are already in Pinecone, so we don't strictly need to load them here
        # But we'll keep the data structure for potential offline use
        embeddings_path = self.settings.embeddings_path
        if embeddings_path.exists():
            logger.info(f"Loading embeddings from: {embeddings_path}")
            with open(embeddings_path, 'rb') as f:
                embeddings_data = pickle.load(f)
                # Extract chunks from the data dictionary
                chunks_with_embeddings = embeddings_data.get('chunks', [])

            # Create chunk_id to embedding mapping
            self.chunk_embeddings = {
                chunk['metadata']['chunk_id']: chunk.get('embedding', [])
                for chunk in chunks_with_embeddings
                if 'embedding' in chunk
            }

            # CRITICAL: Create chunk_id to content mapping
            # Pinecone metadata doesn't include content (to save space)
            # So we need to map chunk_id -> full chunk data including content
            self.chunk_content_map = {
                chunk['metadata']['chunk_id']: chunk.get('content', '')
                for chunk in chunks_with_embeddings
            }

            # CRITICAL: Create chunk_id to full metadata mapping
            # Pinecone metadata is reduced (image_paths list → first_image_path, etc.)
            # We need the full metadata including all image_paths
            self.chunk_metadata_map = {
                chunk['metadata']['chunk_id']: chunk.get('metadata', {})
                for chunk in chunks_with_embeddings
            }

            logger.info(f"Loaded {len(self.chunk_embeddings)} chunk embeddings")
            logger.info(f"Loaded {len(self.chunk_content_map)} chunk contents")
            logger.info(f"Loaded {len(self.chunk_metadata_map)} full metadata entries")
        else:
            logger.warning(f"Embeddings file not found, vector search will rely on Pinecone only")
            self.chunk_embeddings = {}
            self.chunk_content_map = {}
            self.chunk_metadata_map = {}

        # Initialize query expander for synonym-based expansion
        self.query_expander = QueryExpander()

        logger.info("✅ HybridSearch initialized")

    def search(self,
               query: str,
               query_embedding: List[float],
               top_k: Optional[int] = None,
               vector_weight: Optional[float] = None,
               bm25_weight: Optional[float] = None,
               filter_toc: bool = True,
               metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Perform hybrid search with RRF fusion.

        Args:
            query: Query text
            query_embedding: Query embedding vector
            top_k: Number of results to return (default from settings)
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            filter_toc: Filter out table of contents chunks
            metadata_filter: Optional Pinecone metadata filter

        Returns:
            List of ranked chunks with scores
        """
        top_k = top_k or self.settings.vector_top_k
        vector_weight = vector_weight if vector_weight is not None else self.settings.vector_weight
        bm25_weight = bm25_weight if bm25_weight is not None else self.settings.bm25_weight

        logger.info(f"Hybrid search for query: '{query[:50]}...'")
        logger.info(f"  Vector weight: {vector_weight}, BM25 weight: {bm25_weight}")

        # 1. Vector search
        vector_results = self._vector_search(
            query_embedding,
            top_k=self.settings.vector_top_k,
            filter_toc=filter_toc,
            metadata_filter=metadata_filter
        )

        logger.info(f"  Vector search: {len(vector_results)} results")

        # 2. BM25 search
        bm25_results = self._bm25_search(
            query,
            top_k=self.settings.bm25_top_k,
            filter_toc=filter_toc
        )

        logger.info(f"  BM25 search: {len(bm25_results)} results")

        # 3. RRF Fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            k=self.settings.rrf_k
        )

        logger.info(f"  Fused results: {len(fused_results)} unique chunks")

        # 4. Return top-k
        final_results = fused_results[:top_k]

        logger.info(f"✅ Hybrid search complete: {len(final_results)} results")

        return final_results

    def search_with_expansion(self,
                             query: str,
                             query_embedding: List[float],
                             embedding_generator,
                             top_k: Optional[int] = None,
                             max_expansions: int = 2,
                             vector_weight: Optional[float] = None,
                             bm25_weight: Optional[float] = None,
                             filter_toc: bool = True,
                             metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Perform hybrid search with query expansion for better recall.

        Expands the query with synonyms and variations, searches with each,
        then combines and deduplicates results.

        Args:
            query: Query text
            query_embedding: Query embedding vector for original query
            embedding_generator: Generator to create embeddings for expanded queries
            top_k: Number of results to return
            max_expansions: Maximum query variations to try (1-3 recommended)
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search
            filter_toc: Filter out TOC chunks
            metadata_filter: Optional metadata filter

        Returns:
            Combined and ranked results from all query variations
        """
        top_k = top_k or self.settings.vector_top_k
        vector_weight = vector_weight if vector_weight is not None else self.settings.vector_weight
        bm25_weight = bm25_weight if bm25_weight is not None else self.settings.bm25_weight

        # Expand query
        query_variations = self.query_expander.expand_query(query, max_expansions=max_expansions)

        logger.info(f"🔍 Search with expansion: {len(query_variations)} query variations")
        for i, var in enumerate(query_variations, 1):
            logger.debug(f"  {i}. {var}")

        # Search with each variation
        all_results = {}  # chunk_id -> best result
        all_scores = defaultdict(list)  # chunk_id -> list of scores

        for i, query_var in enumerate(query_variations):
            # Use original embedding for first query, generate new for expansions
            if i == 0:
                var_embedding = query_embedding
            else:
                # Generate embedding for expanded query
                var_embedding = embedding_generator.generate_embeddings([query_var])[0]

            # Search with this variation
            results = self.search(
                query=query_var,
                query_embedding=var_embedding,
                top_k=top_k * 2,  # Get more results per variation
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                filter_toc=filter_toc,
                metadata_filter=metadata_filter
            )

            # Collect results
            for result in results:
                chunk_id = result['metadata']['chunk_id']
                score = result.get('score', 0)

                # Track all scores for this chunk
                all_scores[chunk_id].append(score)

                # Keep best result for each chunk
                if chunk_id not in all_results or score > all_results[chunk_id].get('score', 0):
                    all_results[chunk_id] = result

        # Aggregate scores (use max score from all variations)
        for chunk_id in all_results:
            scores = all_scores[chunk_id]
            all_results[chunk_id]['score'] = max(scores)  # Use best score
            all_results[chunk_id]['expansion_hits'] = len(scores)  # How many variations matched

        # Sort by aggregated score
        combined_results = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        # Return top-k
        final_results = combined_results[:top_k]

        logger.info(f"✅ Expansion search complete:")
        logger.info(f"   Query variations: {len(query_variations)}")
        logger.info(f"   Unique chunks found: {len(all_results)}")
        logger.info(f"   Returned: {len(final_results)}")

        return final_results

    def _vector_search(self,
                      query_embedding: List[float],
                      top_k: int,
                      filter_toc: bool = True,
                      metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Perform vector search using Pinecone.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            filter_toc: Filter out TOC chunks
            metadata_filter: Optional metadata filter

        Returns:
            List of results with scores
        """
        # Build filter
        final_filter = metadata_filter or {}
        if filter_toc:
            final_filter['is_toc'] = {'$eq': False}

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=final_filter if final_filter else None
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            chunk_id = match.id

            # CRITICAL FIX: Get content from our chunk_content_map
            # Pinecone metadata doesn't include content (40KB limit)
            content = self.chunk_content_map.get(chunk_id, '')

            if not content:
                logger.warning(f"No content found for chunk_id: {chunk_id}")

            # CRITICAL FIX: Get full metadata from our chunk_metadata_map
            # Pinecone metadata is reduced (image_paths list → first_image_path, etc.)
            full_metadata = self.chunk_metadata_map.get(chunk_id, {})

            # Merge Pinecone metadata with full metadata, preferring full metadata
            # This ensures we get complete image_paths lists and other full fields
            merged_metadata = {**match.metadata, **full_metadata}

            formatted_results.append({
                'chunk_id': chunk_id,
                'score': float(match.score),
                'metadata': merged_metadata,  # ✅ Now using full metadata with complete image_paths
                'content': content,  # ✅ Now using actual content from our mapping
                'source': 'vector'
            })

        return formatted_results

    def _bm25_search(self,
                     query: str,
                     top_k: int,
                     filter_toc: bool = True) -> List[Dict]:
        """
        Perform BM25 keyword search.

        Args:
            query: Query text
            top_k: Number of results
            filter_toc: Filter out TOC chunks

        Returns:
            List of results with scores
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        # Format results
        formatted_results = []
        for idx in top_indices:
            chunk = self.bm25_chunks[idx]
            score = float(scores[idx])

            # Filter TOC if requested
            if filter_toc and chunk['metadata'].get('is_toc', False):
                continue

            # Skip very low scores
            if score < 0.01:
                continue

            formatted_results.append({
                'chunk_id': chunk['metadata']['chunk_id'],
                'score': score,
                'metadata': chunk['metadata'],
                'content': chunk['content'],
                'source': 'bm25'
            })

        return formatted_results

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization matching BM25 index."""
        tokens = text.lower().split()
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def _reciprocal_rank_fusion(self,
                                vector_results: List[Dict],
                                bm25_results: List[Dict],
                                vector_weight: float = 0.5,
                                bm25_weight: float = 0.5,
                                k: int = 60) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF Formula: score = sum(weight / (k + rank))

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            vector_weight: Weight for vector results
            bm25_weight: Weight for BM25 results
            k: RRF constant (typically 60)

        Returns:
            Sorted list of fused results
        """
        # Collect all unique chunks
        all_chunks = {}
        rrf_scores = defaultdict(float)

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result['chunk_id']
            all_chunks[chunk_id] = result
            rrf_scores[chunk_id] += vector_weight / (k + rank)

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result['chunk_id']
            if chunk_id not in all_chunks:
                all_chunks[chunk_id] = result
            rrf_scores[chunk_id] += bm25_weight / (k + rank)

        # Sort by RRF score
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )

        # Build final results
        fused_results = []
        for chunk_id in sorted_chunk_ids:
            chunk = all_chunks[chunk_id]
            chunk['rrf_score'] = rrf_scores[chunk_id]
            chunk['original_score'] = chunk['score']
            chunk['score'] = rrf_scores[chunk_id]
            fused_results.append(chunk)

        return fused_results


if __name__ == "__main__":
    """Test hybrid search."""
    print("\n" + "="*70)
    print("TESTING HYBRID SEARCH")
    print("="*70 + "\n")

    # Initialize
    hybrid_search = HybridSearch()

    # Test query
    test_query = "How do I create a no-code block on Watermelon?"
    print(f"Test Query: {test_query}\n")

    # Generate query embedding
    from src.database.embedding_generator import EmbeddingGenerator
    generator = EmbeddingGenerator()
    query_embedding = generator.generate_embeddings([test_query])[0]

    # Search
    results = hybrid_search.search(
        query=test_query,
        query_embedding=query_embedding,
        top_k=10
    )

    # Display results
    print(f"\n📊 Results ({len(results)} chunks):\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Chunk: {result['chunk_id']}")
        print(f"   RRF Score: {result['rrf_score']:.4f}")
        print(f"   Source: {result.get('source', 'fused')}")
        print(f"   Heading: {' > '.join(result['metadata'].get('heading_path', []))}")
        print(f"   Content: {result['content'][:100]}...")
        print()

    print("✅ Hybrid search test complete!")
