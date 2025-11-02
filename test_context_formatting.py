"""
Test what context is actually being sent to the LLM.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from src.generation.end_to_end_pipeline import EndToEndPipeline

# Initialize pipeline
pipeline = EndToEndPipeline(
    use_reranking=True,
    enable_context_chaining=True,
    validate_responses=True
)

# Test query
query = "How do I set up MS Teams integration?"

print("\n" + "="*70)
print("TESTING CONTEXT FORMATTING")
print("="*70)

# Process query
result = pipeline.process_query(query)

# Check what context was sent
print("\n" + "="*70)
print("CONTEXT SENT TO LLM:")
print("="*70)

# Get the first two chunks
for i, chunk in enumerate(result.retrieval.organized_context.chunks[:2], 1):
    heading = ' > '.join(chunk.get('metadata', {}).get('heading_path', []))
    page = chunk.get('metadata', {}).get('page_start', 0)
    content = chunk.get('content', '')

    print(f"\n### Chunk {i}: {heading} (Page {page})")
    print(f"Content length: {len(content)} chars")
    print(f"Content preview (first 500 chars):")
    print(content[:500])
    print("...")

print("\n" + "="*70)
print("GENERATED ANSWER:")
print("="*70)
print(result.answer.answer[:500])
print("...")
