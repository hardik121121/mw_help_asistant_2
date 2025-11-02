"""
Debug script to investigate why MS Teams integration page is not being retrieved.
"""

import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.database.embedding_generator import EmbeddingGenerator
from src.retrieval.hybrid_search import HybridSearch
from src.query.query_understanding import QueryUnderstandingEngine
from src.retrieval.multi_step_retriever import MultiStepRetriever

print("="*70)
print("MS TEAMS RETRIEVAL DEBUG")
print("="*70)

# The query
query = "How do I set up MS Teams integration?"
print(f"\n📝 Query: {query}\n")

# Step 1: Check if chunks exist
print("="*70)
print("STEP 1: Verify MS Teams chunks exist in dataset")
print("="*70)

chunks_path = Path('cache/hierarchical_chunks_filtered.json')
with open(chunks_path, 'r') as f:
    data = json.load(f)
    chunks = data.get('chunks', [])

ms_teams_chunks = []
for chunk in chunks:
    content = chunk.get('content', '').lower()
    heading_path = chunk.get('metadata', {}).get('heading_path', [])

    if any('ms teams' in h.lower() for h in heading_path):
        ms_teams_chunks.append({
            'chunk_id': chunk.get('metadata', {}).get('chunk_id'),
            'heading_path': ' > '.join(heading_path),
            'page_start': chunk.get('metadata', {}).get('page_start'),
            'content_preview': chunk.get('content', '')[:200]
        })

print(f"\n✅ Found {len(ms_teams_chunks)} MS Teams chunks:\n")
for chunk in ms_teams_chunks:
    print(f"  - {chunk['chunk_id']}: {chunk['heading_path']} (Page {chunk['page_start']})")

# Step 2: Test query understanding
print("\n" + "="*70)
print("STEP 2: Query Understanding & Decomposition")
print("="*70)

query_engine = QueryUnderstandingEngine()
understanding = query_engine.understand(query)

print(f"\nQuery Type: {understanding.classification.query_type.value}")
print(f"Complexity: {understanding.classification.complexity.value}")
print(f"Sub-questions ({len(understanding.decomposition.sub_questions)}):")
for i, sq in enumerate(understanding.decomposition.sub_questions, 1):
    print(f"  {i}. {sq}")

# Step 3: Test hybrid search directly
print("\n" + "="*70)
print("STEP 3: Direct Hybrid Search (Vector + BM25)")
print("="*70)

generator = EmbeddingGenerator()
query_embedding = generator.generate_embeddings([query])[0]

hybrid = HybridSearch()
results = hybrid.search(
    query=query,
    query_embedding=query_embedding,
    top_k=30,
    filter_toc=True
)

print(f"\n📊 Top 30 Hybrid Search Results:\n")
ms_teams_found = False
for i, result in enumerate(results[:30], 1):
    chunk_id = result['chunk_id']
    score = result['score']
    heading = ' > '.join(result['metadata'].get('heading_path', []))
    page = result['metadata'].get('page_start', '?')

    # Highlight MS Teams chunks
    is_ms_teams = 'ms teams' in heading.lower()
    if is_ms_teams:
        ms_teams_found = True
        print(f"🎯 {i}. {chunk_id} (score: {score:.4f}) ⭐ MS TEAMS FOUND!")
    else:
        print(f"   {i}. {chunk_id} (score: {score:.4f})")

    print(f"      {heading} (Page {page})")

if not ms_teams_found:
    print("\n❌ MS Teams chunks NOT found in top 30 hybrid search results!")
else:
    print("\n✅ MS Teams chunks found in hybrid search")

# Step 4: Test full multi-step retrieval
print("\n" + "="*70)
print("STEP 4: Full Multi-Step Retrieval Pipeline")
print("="*70)

retriever = MultiStepRetriever(use_reranking=True, enable_context_chaining=True)
retrieval_result = retriever.retrieve(
    query=query,
    query_understanding=understanding,
    max_chunks=20
)

print(f"\nTotal chunks retrieved: {retrieval_result.total_chunks_retrieved}")
print(f"Final chunks after reranking: {retrieval_result.final_chunks}")
print(f"\n📊 Final Top 20 Chunks:\n")

ms_teams_in_final = False
for i, chunk in enumerate(retrieval_result.organized_context.chunks[:20], 1):
    chunk_id = chunk['metadata']['chunk_id']
    heading = ' > '.join(chunk['metadata'].get('heading_path', []))
    page = chunk['metadata'].get('page_start', '?')

    is_ms_teams = 'ms teams' in heading.lower()
    if is_ms_teams:
        ms_teams_in_final = True
        print(f"🎯 {i}. {chunk_id} ⭐ MS TEAMS FOUND!")
    else:
        print(f"   {i}. {chunk_id}")

    print(f"      {heading} (Page {page})")

if not ms_teams_in_final:
    print("\n❌ MS Teams chunks NOT found in final top 20 results!")
    print("\n🔍 DIAGNOSIS:")
    print("  - The retrieval system is failing to rank MS Teams content highly")
    print("  - Possible causes:")
    print("    1. Query decomposition generating irrelevant sub-questions")
    print("    2. Vector embeddings not capturing 'MS Teams' semantic similarity")
    print("    3. BM25 tokenization issues with 'MS Teams' keyword")
    print("    4. Reranking model (Cohere) ranking irrelevant results higher")
    print("    5. RRF fusion weights favoring wrong results")
else:
    print("\n✅ MS Teams chunks found in final results!")

print("\n" + "="*70)
print("DEBUG COMPLETE")
print("="*70)
