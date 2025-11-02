"""
Phase 5 Orchestrator: Run all embedding and indexing steps.
"""

import sys
import time
from pathlib import Path

print("\n" + "="*70)
print("🚀 PHASE 5: EMBEDDINGS & VECTOR INDEX")
print("="*70)
print("\nThis will:")
print("  1. Generate embeddings for all chunks (~30-45 min, ~$3-5)")
print("  2. Create Pinecone vector index")
print("  3. Upload chunks to Pinecone")
print("  4. Create BM25 keyword index")
print("  5. Validate all indexes")
print("\n" + "="*70 + "\n")

print("Starting Phase 5 automatically...")

# Step 1: Generate embeddings
print("\n" + "="*70)
print("STEP 1/4: GENERATING EMBEDDINGS")
print("="*70 + "\n")

from src.database.embedding_generator import EmbeddingGenerator
from config.settings import get_settings
import json
import pickle

settings = get_settings()

# Load chunks
input_file = settings.cache_dir / "hierarchical_chunks_filtered.json"
if not input_file.exists():
    input_file = settings.chunks_path

print(f"Loading chunks from: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
    chunks = data.get('chunks', [])

print(f"Loaded {len(chunks)} chunks")

# Initialize generator
generator = EmbeddingGenerator(model="text-embedding-3-large")

# Estimate cost
avg_tokens = sum(c['metadata']['token_count'] for c in chunks) / len(chunks)
estimate = generator.estimate_cost(len(chunks), int(avg_tokens))

print(f"\n📊 Cost Estimation:")
print(f"  Chunks: {estimate['num_chunks']:,}")
print(f"  Total tokens: {estimate['total_tokens']:,}")
print(f"  Model: {estimate['model']}")
print(f"  Dimension: {estimate['dimension']}")
print(f"  Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
print(f"  Estimated time: {estimate['estimated_time_minutes']:.1f} minutes")

# Generate embeddings
print(f"\nGenerating embeddings...")
start_time = time.time()
embedded_chunks = generator.generate_embeddings(chunks, show_progress=True)
elapsed_time = time.time() - start_time

print(f"\n⏱️  Time elapsed: {elapsed_time / 60:.1f} minutes")

# Save embeddings
output_file = settings.embeddings_path
generator.save_embeddings(embedded_chunks, output_file)
print(f"✅ Saved embeddings to: {output_file}")

# Step 2 & 3: Create Pinecone index and upload
print("\n" + "="*70)
print("STEP 2/4: CREATING PINECONE INDEX")
print("="*70 + "\n")

from src.database.vector_store import VectorStore

vector_store = VectorStore()

# Create index
print("Creating Pinecone index...")
success = vector_store.create_index()

if not success:
    print("❌ Failed to create index")
    sys.exit(1)

print("\n" + "="*70)
print("STEP 3/4: UPLOADING TO PINECONE")
print("="*70 + "\n")

# Upload chunks
print("Uploading chunks to Pinecone...")
successful, failed = vector_store.upload_chunks(embedded_chunks, show_progress=True)

print(f"\n✅ Upload complete!")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")

# Verify
stats = vector_store.get_index_stats()
print(f"\nIndex statistics:")
print(f"  Total vectors: {stats.get('total_vectors', 0)}")
print(f"  Dimension: {stats.get('dimension', 0)}")
print(f"  Index fullness: {stats.get('index_fullness', 0):.2%}")

# Step 4: Create BM25 index
print("\n" + "="*70)
print("STEP 4/4: CREATING BM25 INDEX")
print("="*70 + "\n")

from src.database.bm25_index import BM25Index

bm25_index = BM25Index()

print("Building BM25 index...")
bm25_index.build_index(chunks)

bm25_stats = bm25_index.get_stats()
print(f"\nBM25 Index Statistics:")
print(f"  Chunks: {bm25_stats['num_chunks']}")
print(f"  Avg document length: {bm25_stats['avg_doc_length']:.1f} tokens")
print(f"  Vocabulary size: {bm25_stats['vocabulary_size']:,}")

# Save BM25 index
bm25_output = settings.cache_dir / "bm25_index.pkl"
bm25_index.save_index(bm25_output)
print(f"✅ Saved BM25 index to: {bm25_output}")

# Final summary
print("\n" + "="*70)
print("🎉 PHASE 5 COMPLETE!")
print("="*70)

print("\n📊 Summary:")
print(f"  ✅ Embeddings generated: {len(embedded_chunks)}")
print(f"  ✅ Pinecone vectors: {stats.get('total_vectors', 0)}")
print(f"  ✅ BM25 chunks: {bm25_stats['num_chunks']}")
print(f"  ⏱️  Total time: {elapsed_time / 60:.1f} minutes")
print(f"  💰 Estimated cost: ${estimate['estimated_cost_usd']:.2f}")

print("\n📁 Created Files:")
print(f"  - {output_file}")
print(f"  - {bm25_output}")
print(f"  - Pinecone index: {vector_store.index_name}")

print("\n🚀 Next Steps:")
print("  - Ready for Phase 4: Multi-Step Retrieval")
print("  - Test retrieval with sample queries")
print("  - Build Phase 6: Generation pipeline")

print("\n" + "="*70 + "\n")
