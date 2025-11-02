"""
BM25 Keyword Search Index.
Creates and manages BM25 index for keyword-based retrieval.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("⚠️  rank-bm25 not installed. Please run: pip install rank-bm25")
    BM25Okapi = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25-based keyword search index.

    Features:
    - Fast keyword matching
    - Relevance scoring
    - Complement to vector search
    - In-memory index (fast queries)
    """

    def __init__(self):
        """Initialize BM25 index."""
        self.settings = get_settings()
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

        logger.info("Initialized BM25Index")

    def build_index(self, chunks: List[Dict]) -> bool:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunks with 'content' field

        Returns:
            True if successful
        """
        if BM25Okapi is None:
            raise RuntimeError("rank-bm25 library not installed")

        logger.info(f"Building BM25 index for {len(chunks)} chunks...")

        # Filter out TOC chunks (optional)
        content_chunks = [
            c for c in chunks
            if not c['metadata'].get('is_toc', False)
        ]

        if len(content_chunks) < len(chunks):
            logger.info(f"Filtered out {len(chunks) - len(content_chunks)} TOC chunks")

        self.chunks = content_chunks

        # Tokenize corpus
        logger.info("Tokenizing corpus...")
        self.tokenized_corpus = [
            self._tokenize(chunk['content'])
            for chunk in content_chunks
        ]

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"✅ BM25 index built with {len(content_chunks)} chunks")
        return True

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Simple whitespace tokenization with lowercasing
        # Could be enhanced with stemming, stopword removal, etc.
        tokens = text.lower().split()

        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]

        return tokens

    def search(self, query: str, top_k: int = 30) -> List[Tuple[Dict, float]]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (chunk, score) tuples
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built yet")

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.chunks[idx], float(scores[idx])))

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def get_top_k(self, query: str, top_k: int = 30) -> List[Dict]:
        """
        Get top-k chunks for query (without scores).

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of chunks
        """
        results = self.search(query, top_k)
        return [chunk for chunk, score in results]

    def save_index(self, output_path: Path):
        """
        Save BM25 index to file.

        Args:
            output_path: Path to save index
        """
        if self.bm25 is None:
            raise RuntimeError("No index to save")

        logger.info(f"Saving BM25 index to: {output_path}")

        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
            'num_chunks': len(self.chunks)
        }

        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved BM25 index ({file_size_mb:.1f} MB)")

    def load_index(self, input_path: Path):
        """
        Load BM25 index from file.

        Args:
            input_path: Path to index file
        """
        logger.info(f"Loading BM25 index from: {input_path}")

        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']

        logger.info(f"✅ Loaded BM25 index with {len(self.chunks)} chunks")

    def get_stats(self) -> Dict:
        """Get statistics about the BM25 index."""
        if self.bm25 is None:
            return {'status': 'not_built'}

        return {
            'status': 'ready',
            'num_chunks': len(self.chunks),
            'avg_doc_length': sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus),
            'vocabulary_size': len(set(token for doc in self.tokenized_corpus for token in doc))
        }


def main():
    """Build BM25 index from chunks."""
    print("\n" + "="*70)
    print("🔎 BM25 INDEX CREATION")
    print("="*70 + "\n")

    settings = get_settings()

    # Load chunks
    input_file = settings.cache_dir / "hierarchical_chunks_filtered.json"
    if not input_file.exists():
        input_file = settings.chunks_path
        logger.warning("Filtered chunks not found, using original chunks")

    print(f"Loading chunks from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])

    print(f"Loaded {len(chunks)} chunks\n")

    # Build index
    print("Building BM25 index...")
    print("-" * 70)

    bm25_index = BM25Index()
    success = bm25_index.build_index(chunks)

    if not success:
        print("❌ Failed to build index")
        return

    # Get stats
    stats = bm25_index.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Avg document length: {stats['avg_doc_length']:.1f} tokens")
    print(f"  Vocabulary size: {stats['vocabulary_size']:,}")

    # Test search
    print("\n" + "-" * 70)
    print("Testing BM25 search...")
    print("-" * 70)

    test_queries = [
        "MS Teams integration",
        "no-code blocks",
        "authentication error"
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = bm25_index.search(query, top_k=3)
        print(f"Found {len(results)} results:")
        for i, (chunk, score) in enumerate(results, 1):
            page = chunk['metadata']['page_start']
            heading = chunk['metadata'].get('current_heading', 'N/A')
            print(f"  {i}. Page {page}, Score: {score:.2f}")
            print(f"     Heading: {heading}")
            print(f"     Preview: {chunk['content'][:80]}...")

    # Save index
    print("\n" + "-" * 70)
    print("Saving BM25 index...")
    print("-" * 70)

    output_file = settings.cache_dir / "bm25_index.pkl"
    bm25_index.save_index(output_file)

    print("\n" + "="*70)
    print("✅ BM25 INDEX CREATION COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_file}")
    print("Ready for hybrid search!\n")


if __name__ == "__main__":
    main()
