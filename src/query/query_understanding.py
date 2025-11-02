"""
Query Understanding Orchestrator.
Coordinates query decomposition, classification, and intent analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

from src.query.query_decomposer import QueryDecomposer, DecomposedQuery
from src.query.query_classifier import QueryClassifier, QueryClassification
from src.query.intent_analyzer import IntentAnalyzer, IntentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class QueryUnderstanding:
    """
    Complete understanding of a query.
    Combines decomposition, classification, and intent analysis.
    """
    original_query: str
    decomposition: DecomposedQuery
    classification: QueryClassification
    intent: IntentAnalysis

    # Derived metadata
    retrieval_strategy: str = "hybrid"  # hybrid, vector_only, keyword_only
    generation_strategy: str = "standard"  # standard, step_by_step, comparison
    priority_topics: List[str] = field(default_factory=list)
    estimated_response_time: float = 5.0  # seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "original_query": self.original_query,
            "decomposition": asdict(self.decomposition),
            "classification": asdict(self.classification),
            "intent": asdict(self.intent),
            "retrieval_strategy": self.retrieval_strategy,
            "generation_strategy": self.generation_strategy,
            "priority_topics": self.priority_topics,
            "estimated_response_time": self.estimated_response_time
        }


class QueryUnderstandingEngine:
    """
    Main orchestrator for query understanding.

    Coordinates all analysis components and produces a complete understanding
    of the user's query for downstream retrieval and generation.
    """

    def __init__(self):
        """Initialize query understanding engine."""
        logger.info("Initializing Query Understanding Engine...")

        self.decomposer = QueryDecomposer()
        self.classifier = QueryClassifier()
        self.analyzer = IntentAnalyzer()

        logger.info("Query Understanding Engine initialized")

    def understand(self, query: str) -> QueryUnderstanding:
        """
        Perform complete query understanding.

        Args:
            query: User's query string

        Returns:
            QueryUnderstanding with all analysis results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Understanding query: {query}")
        logger.info(f"{'='*60}\n")

        # Step 1: Decompose query
        logger.info("Step 1: Decomposing query...")
        decomposition = self.decomposer.decompose(query)
        logger.info(f"  → {len(decomposition.sub_questions)} sub-questions, "
                   f"complexity: {decomposition.query_complexity}")

        # Step 2: Classify query
        logger.info("\nStep 2: Classifying query...")
        topics = [topic for sq in decomposition.sub_questions for topic in sq.topics]
        classification = self.classifier.classify(query, topics)
        logger.info(f"  → Type: {classification.query_type.value}, "
                   f"Complexity: {classification.complexity.value}")

        # Step 3: Analyze intent
        logger.info("\nStep 3: Analyzing intent...")
        intent = self.analyzer.analyze(query)
        logger.info(f"  → Intent: {intent.primary_intent.value}, "
                   f"Entities: {len(intent.entities)}")

        # Step 4: Determine strategies
        logger.info("\nStep 4: Determining strategies...")
        retrieval_strategy = self._determine_retrieval_strategy(
            classification, decomposition
        )
        generation_strategy = self._determine_generation_strategy(
            classification, intent
        )
        logger.info(f"  → Retrieval: {retrieval_strategy}")
        logger.info(f"  → Generation: {generation_strategy}")

        # Step 5: Identify priority topics
        priority_topics = self._identify_priority_topics(
            decomposition, classification, intent
        )

        # Step 6: Estimate response time
        estimated_time = self._estimate_response_time(
            decomposition, classification
        )

        understanding = QueryUnderstanding(
            original_query=query,
            decomposition=decomposition,
            classification=classification,
            intent=intent,
            retrieval_strategy=retrieval_strategy,
            generation_strategy=generation_strategy,
            priority_topics=priority_topics,
            estimated_response_time=estimated_time
        )

        logger.info(f"\n{'='*60}")
        logger.info("Query understanding complete!")
        logger.info(f"{'='*60}\n")

        return understanding

    def _determine_retrieval_strategy(self, classification: QueryClassification,
                                     decomposition: DecomposedQuery) -> str:
        """Determine optimal retrieval strategy."""
        # Complex queries benefit from hybrid search
        if classification.complexity.value in ["complex", "very_complex"]:
            return "hybrid"

        # Conceptual queries work well with vector search
        if classification.query_type.value == "conceptual":
            return "vector_only"

        # Procedural queries need keyword precision
        if classification.query_type.value == "procedural":
            return "hybrid"

        return "hybrid"  # Default

    def _determine_generation_strategy(self, classification: QueryClassification,
                                      intent: IntentAnalysis) -> str:
        """Determine optimal generation strategy."""
        # Step-by-step for procedural queries
        if classification.query_type.value == "procedural":
            return "step_by_step"

        # Comparison format for comparison queries
        if classification.query_type.value == "comparison":
            return "comparison"

        # Troubleshooting format
        if classification.query_type.value == "troubleshooting":
            return "troubleshooting"

        # Code-focused format
        if classification.requires_code:
            return "code_focused"

        return "standard"  # Default

    def _identify_priority_topics(self, decomposition: DecomposedQuery,
                                  classification: QueryClassification,
                                  intent: IntentAnalysis) -> List[str]:
        """Identify priority topics for retrieval."""
        topics = set()

        # Add topics from decomposition (highest priority sub-questions)
        for sq in sorted(decomposition.sub_questions, key=lambda x: x.priority):
            topics.update(sq.topics[:2])  # Top 2 topics per sub-question

        # Add topics from classification
        topics.update(classification.primary_topics[:3])

        # Add entity texts as topics
        for entity in intent.entities[:5]:  # Top 5 entities
            topics.add(entity.text)

        # Return top 10 unique topics
        return list(topics)[:10]

    def _estimate_response_time(self, decomposition: DecomposedQuery,
                               classification: QueryClassification) -> float:
        """Estimate response time in seconds."""
        base_time = 3.0  # Base retrieval + generation time

        # Add time per sub-question
        time_per_subq = 1.5
        base_time += len(decomposition.sub_questions) * time_per_subq

        # Add time based on complexity
        complexity_multipliers = {
            "simple": 1.0,
            "moderate": 1.2,
            "complex": 1.5,
            "very_complex": 2.0
        }
        multiplier = complexity_multipliers.get(classification.complexity.value, 1.0)

        return base_time * multiplier

    def save_understanding(self, understanding: QueryUnderstanding,
                          output_path: Path):
        """Save query understanding to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(understanding.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved understanding to: {output_path}")

    def load_understanding(self, input_path: Path) -> QueryUnderstanding:
        """Load query understanding from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded understanding from: {input_path}")
        return QueryUnderstanding(**data)


def main():
    """Test query understanding engine with various queries."""
    print("\n" + "="*70)
    print("🧠 QUERY UNDERSTANDING ENGINE TEST")
    print("="*70 + "\n")

    engine = QueryUnderstandingEngine()

    # Test queries
    test_queries = [
        "What is MS Teams integration?",
        "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?",
        "Compare Slack integration vs MS Teams integration with Watermelon",
        "Getting authentication error when connecting to Salesforce API"
    ]

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_queries)}")
        print(f"{'='*70}")
        print(f"\n📝 Query: {query}\n")

        understanding = engine.understand(query)
        results.append(understanding)

        # Print summary
        print("\n" + "─"*70)
        print("📊 UNDERSTANDING SUMMARY")
        print("─"*70)

        print(f"\n🔍 Decomposition:")
        print(f"  Complexity: {understanding.decomposition.query_complexity}")
        print(f"  Should Decompose: {understanding.decomposition.should_decompose}")
        print(f"  Sub-questions: {len(understanding.decomposition.sub_questions)}")
        for sq in understanding.decomposition.sub_questions:
            print(f"    • [{sq.id}] {sq.question[:60]}...")

        print(f"\n🏷️  Classification:")
        print(f"  Type: {understanding.classification.query_type.value}")
        print(f"  Complexity: {understanding.classification.complexity.value}")
        print(f"  Expected Format: {understanding.classification.expected_format.value}")
        print(f"  Estimated Chunks: {understanding.classification.estimated_chunks_needed}")

        print(f"\n🎯 Intent:")
        print(f"  Primary: {understanding.intent.primary_intent.value}")
        print(f"  Goal: {understanding.intent.user_goal}")
        print(f"  Entities: {len(understanding.intent.entities)}")
        for entity in understanding.intent.entities[:3]:
            print(f"    • {entity.text} ({entity.type})")

        print(f"\n⚙️  Strategies:")
        print(f"  Retrieval: {understanding.retrieval_strategy}")
        print(f"  Generation: {understanding.generation_strategy}")
        print(f"  Priority Topics: {', '.join(understanding.priority_topics[:5])}")
        print(f"  Estimated Time: {understanding.estimated_response_time:.1f}s")

        print("\n" + "="*70)

    # Save results
    output_dir = Path("tests/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "query_understanding_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)

    print(f"\n✅ Results saved to: {output_file}")
    print("\n" + "="*70)
    print("🎉 QUERY UNDERSTANDING ENGINE TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
