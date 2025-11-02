"""
Pinecone Vector Store Management.
Creates and manages Pinecone index for semantic search.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pickle

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    print("⚠️  Pinecone library not installed. Please run: pip install pinecone-client")
    Pinecone = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages Pinecone vector database for semantic search.

    Features:
    - Index creation and management
    - Batch uploading with metadata
    - Query interface
    - Statistics and validation
    """

    def __init__(self):
        """Initialize Pinecone vector store."""
        self.settings = get_settings()

        if Pinecone is None:
            raise RuntimeError("Pinecone library not installed")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index_name = self.settings.pinecone_index_name
        self.dimension = self.settings.pinecone_dimension
        self.metric = self.settings.pinecone_metric

        logger.info(f"Initialized VectorStore for index: {self.index_name}")

    def create_index(self, dimension: Optional[int] = None,
                    metric: Optional[str] = None) -> bool:
        """
        Create a new Pinecone index.

        Args:
            dimension: Vector dimension (default from settings)
            metric: Distance metric (default from settings)

        Returns:
            True if created successfully
        """
        dimension = dimension or self.dimension
        metric = metric or self.metric

        logger.info(f"Creating index: {self.index_name}")
        logger.info(f"  Dimension: {dimension}")
        logger.info(f"  Metric: {metric}")
        logger.info(f"  Cloud: {self.settings.pinecone_cloud}")
        logger.info(f"  Region: {self.settings.pinecone_region}")

        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            if any(idx.name == self.index_name for idx in existing_indexes):
                logger.warning(f"Index '{self.index_name}' already exists")
                response = input("Delete and recreate? (yes/no): ")
                if response.lower() in ['yes', 'y']:
                    self.pc.delete_index(self.index_name)
                    logger.info(f"Deleted existing index: {self.index_name}")
                    time.sleep(5)  # Wait for deletion
                else:
                    logger.info("Using existing index")
                    return True

            # Create serverless index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region
                )
            )

            logger.info(f"✅ Created index: {self.index_name}")
            logger.info("Waiting for index to be ready...")

            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

            logger.info("✅ Index is ready!")
            return True

        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False

    def upload_chunks(self, chunks: List[Dict], batch_size: Optional[int] = None,
                     show_progress: bool = True) -> Tuple[int, int]:
        """
        Upload chunks with embeddings to Pinecone.

        Args:
            chunks: Chunks with 'embedding' and 'metadata' fields
            batch_size: Batch size for uploads (default from settings)
            show_progress: Show progress bar

        Returns:
            Tuple of (successful_uploads, failed_uploads)
        """
        batch_size = batch_size or self.settings.pinecone_batch_size

        # Get index
        index = self.pc.Index(self.index_name)

        logger.info(f"Uploading {len(chunks)} chunks to Pinecone...")
        logger.info(f"Batch size: {batch_size}")

        # Filter chunks with embeddings
        chunks_with_embeddings = [
            c for c in chunks if 'embedding' in c
        ]

        if len(chunks_with_embeddings) < len(chunks):
            logger.warning(
                f"Only {len(chunks_with_embeddings)}/{len(chunks)} chunks have embeddings"
            )

        successful = 0
        failed = 0

        # Process in batches
        total_batches = (len(chunks_with_embeddings) + batch_size - 1) // batch_size

        iterator = range(0, len(chunks_with_embeddings), batch_size)
        if show_progress and tqdm:
            iterator = tqdm(iterator, total=total_batches, desc="Uploading to Pinecone")

        for i in iterator:
            batch = chunks_with_embeddings[i:i + batch_size]

            try:
                # Prepare vectors for Pinecone
                vectors = []
                for chunk in batch:
                    # Prepare metadata (Pinecone has limits on metadata size)
                    metadata = self._prepare_metadata(chunk['metadata'])

                    vectors.append({
                        'id': chunk['metadata']['chunk_id'],
                        'values': chunk['embedding'],
                        'metadata': metadata
                    })

                # Upsert to Pinecone
                index.upsert(vectors=vectors)
                successful += len(batch)

            except Exception as e:
                logger.error(f"Error uploading batch {i}: {e}")
                failed += len(batch)

        logger.info(f"\n✅ Upload complete!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")

        return successful, failed

    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """
        Prepare metadata for Pinecone (remove non-serializable fields, reduce size).

        Pinecone limitations:
        - Max metadata size: 40KB per vector
        - Must be JSON-serializable
        """
        # Select important fields only
        pinecone_metadata = {
            'chunk_id': metadata.get('chunk_id', ''),
            'page_start': metadata.get('page_start', 0),
            'page_end': metadata.get('page_end', 0),
            'section_id': metadata.get('section_id', ''),

            # Hierarchy (keep first 3 levels only)
            'heading_path': metadata.get('heading_path', [])[:3],
            'current_heading': metadata.get('current_heading', ''),
            'heading_level': metadata.get('heading_level', 0),

            # Content characteristics
            'content_type': metadata.get('content_type', 'mixed'),
            'technical_depth': metadata.get('technical_depth', 'medium'),

            # Flags
            'has_images': metadata.get('has_images', False),
            'has_tables': metadata.get('has_tables', False),
            'has_code': metadata.get('has_code', False),
            'has_lists': metadata.get('has_lists', False),
            'is_toc': metadata.get('is_toc', False),

            # Keep first image path only
            'first_image_path': metadata.get('image_paths', [''])[0] if metadata.get('image_paths') else '',

            # Size
            'token_count': metadata.get('token_count', 0),
            'char_count': metadata.get('char_count', 0),
        }

        return pinecone_metadata

    def get_index_stats(self) -> Dict:
        """Get statistics about the Pinecone index."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()

            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }

        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

    def search(self, query_embedding: List[float], top_k: int = 10,
              filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filter_dict: Metadata filter (e.g., {"is_toc": False})

        Returns:
            List of matching chunks with scores
        """
        try:
            index = self.pc.Index(self.index_name)

            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            matches = []
            for match in results.matches:
                matches.append({
                    'chunk_id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                })

            return matches

        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    def delete_index(self) -> bool:
        """Delete the Pinecone index."""
        try:
            logger.warning(f"Deleting index: {self.index_name}")
            self.pc.delete_index(self.index_name)
            logger.info("✅ Index deleted")
            return True

        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False


def main():
    """Create Pinecone index and upload chunks."""
    print("\n" + "="*70)
    print("📌 PINECONE VECTOR STORE SETUP")
    print("="*70 + "\n")

    settings = get_settings()
    vector_store = VectorStore()

    # Step 1: Create index
    print("Step 1: Creating Pinecone index...")
    print("-" * 70)
    success = vector_store.create_index()

    if not success:
        print("❌ Failed to create index")
        return

    print()

    # Step 2: Load embeddings
    print("Step 2: Loading embeddings...")
    print("-" * 70)

    embeddings_file = settings.embeddings_path
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        print("Please run embedding_generator first:")
        print("  python -m src.database.embedding_generator")
        return

    print(f"Loading from: {embeddings_file}")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
        chunks = data['chunks']

    print(f"Loaded {len(chunks)} chunks")
    chunks_with_embeddings = sum(1 for c in chunks if 'embedding' in c)
    print(f"Chunks with embeddings: {chunks_with_embeddings}\n")

    # Step 3: Upload to Pinecone
    print("Step 3: Uploading to Pinecone...")
    print("-" * 70)

    successful, failed = vector_store.upload_chunks(chunks, show_progress=True)

    print()

    # Step 4: Verify
    print("Step 4: Verifying upload...")
    print("-" * 70)

    stats = vector_store.get_index_stats()
    print(f"Index statistics:")
    print(f"  Total vectors: {stats.get('total_vectors', 0)}")
    print(f"  Dimension: {stats.get('dimension', 0)}")
    print(f"  Index fullness: {stats.get('index_fullness', 0):.2%}")

    print("\n" + "="*70)
    print("✅ PINECONE SETUP COMPLETE")
    print("="*70)
    print(f"\nIndex: {vector_store.index_name}")
    print(f"Vectors: {stats.get('total_vectors', 0)}")
    print("Ready for retrieval!\n")


if __name__ == "__main__":
    main()
