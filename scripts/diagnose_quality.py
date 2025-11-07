"""
Diagnostic script to identify quality issues in the RAG pipeline.

Usage: python scripts/diagnose_quality.py "Your test query here"
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.end_to_end_pipeline import EndToEndPipeline
from src.query.query_understanding import QueryUnderstandingEngine
from src.retrieval.multi_step_retriever import MultiStepRetriever


def diagnose_query(query: str):
    """Run full diagnostic on a query."""

    print("\n" + "="*80)
    print("🔍 RAG PIPELINE DIAGNOSTIC")
    print("="*80)
    print(f"Query: {query}")
    print("="*80)

    # Step 1: Query Understanding
    print("\n📋 STEP 1: Query Understanding")
    print("-"*80)
    query_engine = QueryUnderstandingEngine()
    understanding = query_engine.understand(query)

    print(f"Classification: {understanding.classification.query_type}")
    print(f"Complexity: {understanding.classification.complexity}")
    print(f"Number of sub-questions: {len(understanding.decomposition.sub_questions)}")
    print(f"\nSub-questions:")
    for i, sq in enumerate(understanding.decomposition.sub_questions, 1):
        print(f"  {i}. {sq}")

    # Step 2: Retrieval
    print("\n🔍 STEP 2: Retrieval Analysis")
    print("-"*80)
    retriever = MultiStepRetriever()
    retrieval_result = retriever.retrieve(query, understanding)

    chunks = retrieval_result.organized_context.chunks
    print(f"Total chunks retrieved: {len(chunks)}")

    # Calculate average score
    scores = [c.get('score', 0) for c in chunks if 'score' in c]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average score: {avg_score:.3f}")

    print(f"Total characters: {sum(len(c.get('content', '')) for c in chunks)}")

    # Check for empty content
    empty_chunks = [c for c in chunks if not c.get('content', '').strip()]
    if empty_chunks:
        print(f"\n⚠️  WARNING: {len(empty_chunks)} chunks have EMPTY CONTENT!")

    # Show top 5 chunks
    print(f"\n📑 Top 5 Retrieved Chunks:")
    for i, chunk in enumerate(chunks[:5], 1):
        content_preview = chunk.get('content', '')[:150].replace('\n', ' ')
        score = chunk.get('score', 0)
        heading = chunk.get('metadata', {}).get('current_heading', 'Unknown')
        page = chunk.get('metadata', {}).get('page_start', '?')
        has_images = chunk.get('metadata', {}).get('has_images', False)

        print(f"\n  {i}. Score: {score:.3f} | Page: {page} | Images: {'✓' if has_images else '✗'}")
        print(f"     Heading: {heading}")
        print(f"     Content: {content_preview}...")

    # Check coverage
    print(f"\n🎯 Coverage Analysis:")
    covered_topics = set()
    for chunk in chunks:
        heading_path = chunk.get('metadata', {}).get('heading_path', [])
        if heading_path:
            covered_topics.add(heading_path[0] if heading_path else 'Unknown')

    print(f"  Distinct top-level topics covered: {len(covered_topics)}")
    print(f"  Topics: {', '.join(list(covered_topics)[:5])}")

    # Step 3: Generation
    print("\n✨ STEP 3: Answer Generation")
    print("-"*80)
    pipeline = EndToEndPipeline()
    result = pipeline.process_query(query)

    print(f"Answer length: {len(result.answer.answer)} characters")
    print(f"Word count: {len(result.answer.answer.split())}")
    print(f"Overall quality score: {result.validation.overall_score:.3f}")
    print(f"Completeness score: {result.validation.completeness_score:.3f}")

    # Show answer preview
    print(f"\n📝 Answer Preview (first 500 chars):")
    print("-"*80)
    print(result.answer.answer[:500])
    print("-"*80)

    # Step 4: Identify Issues
    print("\n🚨 ISSUE IDENTIFICATION")
    print("-"*80)

    issues = []
    suggestions = []

    # Check retrieval issues
    if len(chunks) < 10:
        issues.append(f"Low retrieval count ({len(chunks)} chunks)")
        suggestions.append("Increase vector_top_k and bm25_top_k in config/settings.py")

    if empty_chunks:
        issues.append(f"{len(empty_chunks)} chunks have empty content")
        suggestions.append("Check hybrid_search.py content mapping - see MS_TEAMS_INTEGRATION_FIX.md")

    if avg_score < 0.7:
        issues.append(f"Low average retrieval score ({avg_score:.3f})")
        suggestions.append("Consider query expansion or embedding fine-tuning")

    if len(understanding.decomposition.sub_questions) < 2 and understanding.classification.complexity == "high":
        issues.append("High complexity query but only 1 sub-question generated")
        suggestions.append("Improve query decomposition prompts")

    if len(covered_topics) < 2 and len(understanding.decomposition.sub_questions) > 2:
        issues.append("Multi-topic query but retrieval only covers 1 topic")
        suggestions.append("Check if chunks are properly tagged with heading_path metadata")

    total_content = sum(len(c.get('content', '')) for c in chunks)
    if total_content < 2000:
        issues.append(f"Very little context retrieved ({total_content} chars)")
        suggestions.append("Check if chunks exist in cache/hierarchical_chunks_filtered.json")

    if result.validation.overall_score < 0.7:
        issues.append(f"Low generation quality ({result.validation.overall_score:.3f})")
        suggestions.append("Review answer - may indicate insufficient context")

    if not issues:
        print("✅ No major issues detected! System is working well.")
    else:
        print("Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print(f"\n💡 Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80 + "\n")

    return {
        'understanding': understanding,
        'retrieval': retrieval_result,
        'answer': result,
        'issues': issues,
        'suggestions': suggestions
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_quality.py \"Your test query here\"")
        print("\nExample:")
        print('  python scripts/diagnose_quality.py "How do I integrate MS Teams with Watermelon?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    diagnose_query(query)
