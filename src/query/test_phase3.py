"""
Test Phase 3 Query Understanding with actual test queries.
"""

import json
import logging
from pathlib import Path

from src.query.query_understanding import QueryUnderstandingEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_queries(file_path: str = "tests/test_queries.json") -> list:
    """Load test queries from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get("queries", [])


def main():
    """Test with actual complex queries from test dataset."""
    print("\n" + "="*70)
    print("🧪 PHASE 3 TEST: QUERY UNDERSTANDING WITH COMPLEX QUERIES")
    print("="*70 + "\n")

    # Load test queries
    test_queries = load_test_queries()
    print(f"Loaded {len(test_queries)} test queries\n")

    # Initialize engine
    engine = QueryUnderstandingEngine()

    # Test first 5 complex queries
    results = []
    for i, query_data in enumerate(test_queries[:5], 1):
        query = query_data["query"]

        print(f"\n{'='*70}")
        print(f"TEST {i}/5")
        print(f"{'='*70}")
        print(f"\n📝 Query: {query}")
        print(f"   Type: {query_data['type']}")
        print(f"   Complexity: {query_data['complexity']}")
        print(f"   Expected Topics: {', '.join(query_data['topics'])}\n")

        # Understand query
        understanding = engine.understand(query)
        results.append({
            "test_id": query_data["id"],
            "query": query,
            "test_metadata": query_data,
            "understanding": understanding.to_dict()
        })

        # Print comparison
        print("\n" + "─"*70)
        print("📊 RESULTS")
        print("─"*70)

        print(f"\n✓ Decomposition:")
        print(f"  Complexity: {understanding.decomposition.query_complexity}")
        print(f"  Sub-questions: {len(understanding.decomposition.sub_questions)}")
        if understanding.decomposition.reasoning:
            print(f"  Reasoning: {understanding.decomposition.reasoning[:100]}...")

        for sq in understanding.decomposition.sub_questions:
            print(f"\n  [{sq.id}] {sq.question}")
            print(f"      Topics: {', '.join(sq.topics)}")
            print(f"      Priority: {sq.priority}")

        print(f"\n✓ Classification:")
        print(f"  Type: {understanding.classification.query_type.value}")
        print(f"  Complexity: {understanding.classification.complexity.value}")
        print(f"  Expected Format: {understanding.classification.expected_format.value}")
        print(f"  Technical Depth: {understanding.classification.technical_depth}")

        print(f"\n✓ Intent:")
        print(f"  Primary Intent: {understanding.intent.primary_intent.value}")
        print(f"  User Goal: {understanding.intent.user_goal}")
        print(f"  Expected Outcome: {understanding.intent.expected_outcome}")

        if understanding.intent.entities:
            print(f"\n  Entities:")
            for entity in understanding.intent.entities[:5]:
                print(f"    • {entity.text} ({entity.type})")

        print(f"\n✓ Strategies:")
        print(f"  Retrieval: {understanding.retrieval_strategy}")
        print(f"  Generation: {understanding.generation_strategy}")
        print(f"  Est. Response Time: {understanding.estimated_response_time:.1f}s")

        print(f"\n✓ Topic Comparison:")
        print(f"  Expected: {', '.join(query_data['topics'])}")
        print(f"  Detected: {', '.join(understanding.priority_topics[:5])}")

        print("\n" + "="*70)

    # Save results
    output_dir = Path("tests/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "phase3_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✅ Test results saved to: {output_file}")

    # Generate summary report
    print("\n" + "="*70)
    print("📊 PHASE 3 TEST SUMMARY")
    print("="*70)

    total = len(results)
    decomposed = sum(1 for r in results if r["understanding"]["decomposition"]["should_decompose"])

    print(f"\nTotal Queries Tested: {total}")
    print(f"Complex Queries Decomposed: {decomposed}/{total}")
    print(f"Average Sub-questions: {sum(len(r['understanding']['decomposition']['sub_questions']) for r in results) / total:.1f}")

    complexity_dist = {}
    for r in results:
        comp = r["understanding"]["classification"]["complexity"]
        complexity_dist[comp] = complexity_dist.get(comp, 0) + 1

    print(f"\nComplexity Distribution:")
    for complexity, count in sorted(complexity_dist.items()):
        print(f"  {complexity}: {count}")

    intent_dist = {}
    for r in results:
        intent = r["understanding"]["intent"]["primary_intent"]
        intent_dist[intent] = intent_dist.get(intent, 0) + 1

    print(f"\nIntent Distribution:")
    for intent, count in sorted(intent_dist.items()):
        print(f"  {intent}: {count}")

    print("\n" + "="*70)
    print("✅ PHASE 3 QUERY UNDERSTANDING: COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
