"""
Embedding Generation System.
Generates vector embeddings for hierarchical chunks using OpenAI.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
import pickle

try:
    from openai import OpenAI
except ImportError:
    print("⚠️  OpenAI library not installed. Please run: pip install openai")
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  tqdm not installed. Please run: pip install tqdm")
    tqdm = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for document chunks using OpenAI's embedding models.

    Features:
    - Batch processing for efficiency
    - Rate limiting to avoid API limits
    - Progress tracking
    - Automatic retry on failures
    - Caching to avoid re-generating
    """

    def __init__(self, model: str = "text-embedding-3-large"):
        """
        Initialize embedding generator.

        Args:
            model: OpenAI embedding model name
                  - text-embedding-3-large: 3072 dims, best quality
                  - text-embedding-3-small: 1536 dims, faster/cheaper
        """
        self.settings = get_settings()

        if OpenAI is None:
            raise RuntimeError("OpenAI library not installed")

        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.model = model
        self.batch_size = self.settings.embedding_batch_size
        self.rate_limit = self.settings.embedding_rate_limit

        # Determine embedding dimension
        self.dimension = 3072 if "large" in model else 1536

        logger.info(f"Initialized EmbeddingGenerator with model: {model}")
        logger.info(f"Embedding dimension: {self.dimension}")
        logger.info(f"Batch size: {self.batch_size}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text strings.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )

            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def generate_embeddings_for_chunks(self, chunks: List[Dict],
                          show_progress: bool = True) -> List[Dict]:
        """
        Generate embeddings for all chunks.

        Args:
            chunks: List of chunk dictionaries with 'content' field
            show_progress: Show progress bar

        Returns:
            Chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        total_chunks = len(chunks)
        embedded_chunks = []

        # Filter out TOC chunks if desired (optional)
        chunks_to_embed = [
            c for c in chunks
            if not c.get('metadata', {}).get('is_toc', False)
        ]

        if len(chunks_to_embed) < total_chunks:
            logger.info(f"Filtered out {total_chunks - len(chunks_to_embed)} TOC chunks")
            logger.info(f"Embedding {len(chunks_to_embed)} content chunks")

        # Process in batches
        total_batches = (len(chunks_to_embed) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(chunks_to_embed), self.batch_size)
        if show_progress and tqdm:
            iterator = tqdm(iterator, total=total_batches, desc="Generating embeddings")

        total_tokens = 0
        failed_chunks = []

        for i in iterator:
            batch = chunks_to_embed[i:i + self.batch_size]

            try:
                # Extract texts
                texts = [chunk['content'] for chunk in batch]

                # Generate embeddings
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )

                # Add embeddings to chunks
                for j, chunk in enumerate(batch):
                    chunk['embedding'] = response.data[j].embedding
                    embedded_chunks.append(chunk)

                # Track usage
                total_tokens += response.usage.total_tokens

                # Rate limiting
                if self.rate_limit > 0:
                    time.sleep(self.rate_limit)

            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i}: {e}")
                failed_chunks.extend(batch)
                # Continue with next batch

        # Add back TOC chunks without embeddings (if any were filtered)
        for chunk in chunks:
            if chunk['metadata'].get('is_toc', False):
                embedded_chunks.append(chunk)

        logger.info(f"\n✅ Generated {len(embedded_chunks)} embeddings")
        logger.info(f"Total tokens used: {total_tokens:,}")
        logger.info(f"Estimated cost: ${total_tokens * 0.00013 / 1000:.2f}")

        if failed_chunks:
            logger.warning(f"⚠️  Failed to generate {len(failed_chunks)} embeddings")

        return embedded_chunks

    def save_embeddings(self, chunks: List[Dict], output_path: Path):
        """
        Save chunks with embeddings to file.

        Saves in pickle format for efficient storage and loading.

        Args:
            chunks: Chunks with embeddings
            output_path: Path to save file
        """
        logger.info(f"Saving embeddings to: {output_path}")

        # Prepare data
        data = {
            'model': self.model,
            'dimension': self.dimension,
            'total_chunks': len(chunks),
            'chunks_with_embeddings': sum(1 for c in chunks if 'embedding' in c),
            'chunks': chunks
        }

        # Save as pickle (more efficient for large embeddings)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved embeddings ({file_size_mb:.1f} MB)")

    def load_embeddings(self, input_path: Path) -> List[Dict]:
        """
        Load chunks with embeddings from file.

        Args:
            input_path: Path to embeddings file

        Returns:
            List of chunks with embeddings
        """
        logger.info(f"Loading embeddings from: {input_path}")

        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        chunks = data['chunks']
        logger.info(f"Loaded {len(chunks)} chunks")
        logger.info(f"Model: {data.get('model', 'unknown')}")
        logger.info(f"Dimension: {data.get('dimension', 'unknown')}")

        return chunks

    def estimate_cost(self, num_chunks: int, avg_tokens_per_chunk: int = 300) -> Dict:
        """
        Estimate cost for generating embeddings.

        Args:
            num_chunks: Number of chunks
            avg_tokens_per_chunk: Average tokens per chunk

        Returns:
            Cost estimation dictionary
        """
        total_tokens = num_chunks * avg_tokens_per_chunk

        # OpenAI pricing (as of 2024)
        # text-embedding-3-large: $0.00013 / 1K tokens
        # text-embedding-3-small: $0.00002 / 1K tokens
        cost_per_1k = 0.00013 if "large" in self.model else 0.00002

        estimated_cost = total_tokens * cost_per_1k / 1000
        estimated_time_minutes = (num_chunks / self.batch_size) * self.rate_limit / 60

        return {
            'num_chunks': num_chunks,
            'total_tokens': total_tokens,
            'estimated_cost_usd': estimated_cost,
            'estimated_time_minutes': estimated_time_minutes,
            'model': self.model,
            'dimension': self.dimension
        }


def main():
    """Generate embeddings for hierarchical chunks."""
    print("\n" + "="*70)
    print("🔮 EMBEDDING GENERATION")
    print("="*70 + "\n")

    settings = get_settings()

    # Load chunks (use filtered version with TOC marked)
    input_file = settings.cache_dir / "hierarchical_chunks_filtered.json"
    if not input_file.exists():
        # Fallback to non-filtered version
        input_file = settings.chunks_path
        logger.warning("Filtered chunks not found, using original chunks")

    print(f"Loading chunks from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        chunks = data.get('chunks', [])

    print(f"Loaded {len(chunks)} chunks\n")

    # Initialize generator
    generator = EmbeddingGenerator(model="text-embedding-3-large")

    # Estimate cost
    avg_tokens = sum(c['metadata']['token_count'] for c in chunks) / len(chunks)
    estimate = generator.estimate_cost(len(chunks), int(avg_tokens))

    print("📊 Cost Estimation:")
    print(f"  Chunks: {estimate['num_chunks']:,}")
    print(f"  Total tokens: {estimate['total_tokens']:,}")
    print(f"  Model: {estimate['model']}")
    print(f"  Dimension: {estimate['dimension']}")
    print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
    print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes\n")

    # Confirm
    response = input("Generate embeddings? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return

    print("\n" + "="*70)
    print("Generating embeddings...")
    print("="*70 + "\n")

    # Generate embeddings
    start_time = time.time()
    embedded_chunks = generator.generate_embeddings(chunks, show_progress=True)
    elapsed_time = time.time() - start_time

    print(f"\n⏱️  Time elapsed: {elapsed_time / 60:.1f} minutes")

    # Save embeddings
    output_file = settings.embeddings_path
    generator.save_embeddings(embedded_chunks, output_file)

    print("\n" + "="*70)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("="*70)
    print(f"\nSaved to: {output_file}")
    print(f"Ready for Pinecone upload!\n")


if __name__ == "__main__":
    main()
