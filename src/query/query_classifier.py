"""
Query Classification System.
Classifies queries by type, complexity, and required retrieval strategy.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries."""
    SINGLE_TOPIC = "single_topic"
    MULTI_TOPIC = "multi_topic"
    PROCEDURAL = "procedural"  # How-to questions
    CONCEPTUAL = "conceptual"  # What-is questions
    TROUBLESHOOTING = "troubleshooting"  # Debug/fix questions
    COMPARISON = "comparison"  # Compare A vs B
    INTEGRATION = "integration"  # Connect multiple systems
    SECURITY = "security"  # Security/compliance questions


class ComplexityLevel(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"  # 1 topic, straightforward
    MODERATE = "moderate"  # 2 topics, some context needed
    COMPLEX = "complex"  # 3-4 topics, multi-step reasoning
    VERY_COMPLEX = "very_complex"  # 5+ topics, extensive context


class ResponseFormat(str, Enum):
    """Expected response format."""
    SHORT_ANSWER = "short_answer"  # 1-2 sentences
    PARAGRAPH = "paragraph"  # 1-2 paragraphs
    STEP_BY_STEP = "step_by_step"  # Numbered instructions
    DETAILED_GUIDE = "detailed_guide"  # Comprehensive explanation
    CODE_EXAMPLE = "code_example"  # With code snippets
    COMPARISON_TABLE = "comparison_table"  # Side-by-side comparison


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    complexity: ComplexityLevel
    topic_count: int
    primary_topics: List[str]
    expected_format: ResponseFormat
    requires_images: bool = False
    requires_code: bool = False
    requires_tables: bool = False
    technical_depth: str = "medium"  # low, medium, high
    estimated_chunks_needed: int = 5
    confidence: float = 0.8  # Classification confidence


class QueryClassifier:
    """
    Classifies queries to determine optimal retrieval and generation strategy.

    Uses rule-based heuristics and keyword matching for fast, accurate classification.
    """

    def __init__(self):
        """Initialize query classifier."""
        logger.info("Initialized QueryClassifier")

        # Keywords for different query types
        self.procedural_keywords = [
            "how to", "how do i", "steps to", "guide to", "tutorial",
            "create", "setup", "configure", "install", "deploy", "implement"
        ]

        self.conceptual_keywords = [
            "what is", "what are", "explain", "describe", "define",
            "introduction to", "overview of", "understanding"
        ]

        self.troubleshooting_keywords = [
            "error", "issue", "problem", "not working", "fails", "fix",
            "debug", "troubleshoot", "resolve", "why doesn't"
        ]

        self.comparison_keywords = [
            "vs", "versus", "compare", "difference between", "which",
            "better", "pros and cons", "advantages"
        ]

        self.integration_keywords = [
            "integrate", "integration", "connect", "link", "sync",
            "api", "webhook", "third-party", "external"
        ]

        self.security_keywords = [
            "security", "authentication", "authorization", "permission",
            "access", "encryption", "compliance", "privacy", "gdpr"
        ]

        self.technical_keywords = [
            "api", "code", "function", "method", "parameter", "endpoint",
            "query", "database", "algorithm", "architecture"
        ]

    def classify(self, query: str, topics: Optional[List[str]] = None) -> QueryClassification:
        """
        Classify a query.

        Args:
            query: User's query string
            topics: Optional list of identified topics

        Returns:
            QueryClassification with type, complexity, and metadata
        """
        logger.info(f"Classifying query: {query}")

        query_lower = query.lower()

        # Determine query type
        query_type = self._determine_type(query_lower)

        # Determine complexity
        complexity = self._determine_complexity(query_lower, topics)

        # Determine topic count
        topic_count = len(topics) if topics else self._estimate_topic_count(query_lower)

        # Extract primary topics
        primary_topics = topics[:3] if topics else self._extract_primary_topics(query_lower)

        # Determine expected response format
        expected_format = self._determine_format(query_lower, query_type)

        # Check if special content is needed
        requires_images = self._requires_images(query_lower)
        requires_code = self._requires_code(query_lower)
        requires_tables = self._requires_tables(query_lower)

        # Determine technical depth
        technical_depth = self._determine_technical_depth(query_lower)

        # Estimate chunks needed
        estimated_chunks = self._estimate_chunks_needed(complexity, topic_count)

        classification = QueryClassification(
            query_type=query_type,
            complexity=complexity,
            topic_count=topic_count,
            primary_topics=primary_topics,
            expected_format=expected_format,
            requires_images=requires_images,
            requires_code=requires_code,
            requires_tables=requires_tables,
            technical_depth=technical_depth,
            estimated_chunks_needed=estimated_chunks,
            confidence=0.85
        )

        logger.info(f"Classification: {query_type.value}, complexity: {complexity.value}")
        return classification

    def _determine_type(self, query: str) -> QueryType:
        """Determine query type based on keywords."""
        # Check each type in priority order
        if any(kw in query for kw in self.security_keywords):
            return QueryType.SECURITY

        if any(kw in query for kw in self.troubleshooting_keywords):
            return QueryType.TROUBLESHOOTING

        if any(kw in query for kw in self.comparison_keywords):
            return QueryType.COMPARISON

        if any(kw in query for kw in self.integration_keywords):
            return QueryType.INTEGRATION

        if any(kw in query for kw in self.procedural_keywords):
            return QueryType.PROCEDURAL

        if any(kw in query for kw in self.conceptual_keywords):
            return QueryType.CONCEPTUAL

        # Check for multiple topics
        and_count = query.count(" and ")
        or_count = query.count(" or ")
        if and_count + or_count >= 2:
            return QueryType.MULTI_TOPIC

        return QueryType.SINGLE_TOPIC

    def _determine_complexity(self, query: str, topics: Optional[List[str]]) -> ComplexityLevel:
        """Determine complexity level."""
        # Word count is a good indicator
        word_count = len(query.split())

        # Topic count
        topic_count = len(topics) if topics else self._estimate_topic_count(query)

        # Conjunction count
        conjunctions = query.count(" and ") + query.count(" then ") + query.count(" also ")

        # Determine complexity
        if word_count < 10 and topic_count <= 1:
            return ComplexityLevel.SIMPLE
        elif word_count < 20 and topic_count <= 2 and conjunctions <= 1:
            return ComplexityLevel.MODERATE
        elif word_count < 40 and topic_count <= 4 and conjunctions <= 3:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX

    def _estimate_topic_count(self, query: str) -> int:
        """Estimate number of topics in query."""
        # Count conjunctions as proxy for topics
        and_count = query.count(" and ")
        or_count = query.count(" or ")
        then_count = query.count(" then ")

        # Count capitalized terms (likely product/feature names)
        words = query.split()
        cap_count = sum(1 for w in words if w and w[0].isupper() and len(w) > 2)

        # Estimate
        estimated = 1 + and_count + or_count + then_count + (cap_count // 3)
        return min(estimated, 6)  # Cap at 6

    def _extract_primary_topics(self, query: str) -> List[str]:
        """Extract primary topics from query."""
        topics = []

        # Look for capitalized terms
        words = query.split()
        current_phrase = []

        for word in words:
            clean = word.strip(",.?!:;")
            if clean and len(clean) > 2 and clean[0].isupper():
                current_phrase.append(clean)
            else:
                if len(current_phrase) >= 1:
                    topics.append(" ".join(current_phrase))
                current_phrase = []

        if current_phrase:
            topics.append(" ".join(current_phrase))

        # Add technical terms
        tech_terms = ["api", "integration", "authentication", "workflow",
                     "testing", "deployment", "configuration"]

        for term in tech_terms:
            if term in query:
                topics.append(term)

        return list(set(topics))[:3]  # Top 3 unique topics

    def _determine_format(self, query: str, query_type: QueryType) -> ResponseFormat:
        """Determine expected response format."""
        if query_type == QueryType.PROCEDURAL:
            if "step" in query or "guide" in query:
                return ResponseFormat.STEP_BY_STEP
            return ResponseFormat.DETAILED_GUIDE

        elif query_type == QueryType.CONCEPTUAL:
            if "brief" in query or "quick" in query:
                return ResponseFormat.SHORT_ANSWER
            return ResponseFormat.PARAGRAPH

        elif query_type == QueryType.COMPARISON:
            return ResponseFormat.COMPARISON_TABLE

        elif query_type == QueryType.TROUBLESHOOTING:
            return ResponseFormat.STEP_BY_STEP

        elif "code" in query or "example" in query:
            return ResponseFormat.CODE_EXAMPLE

        return ResponseFormat.DETAILED_GUIDE

    def _requires_images(self, query: str) -> bool:
        """Check if query likely needs images."""
        image_keywords = ["screenshot", "image", "diagram", "visual", "ui", "interface"]
        return any(kw in query for kw in image_keywords)

    def _requires_code(self, query: str) -> bool:
        """Check if query likely needs code examples."""
        code_keywords = ["code", "api", "function", "method", "script",
                        "example", "implement", "snippet"]
        return any(kw in query for kw in code_keywords)

    def _requires_tables(self, query: str) -> bool:
        """Check if query likely needs tables."""
        table_keywords = ["compare", "comparison", "versus", "vs", "table",
                         "list", "options"]
        return any(kw in query for kw in table_keywords)

    def _determine_technical_depth(self, query: str) -> str:
        """Determine required technical depth."""
        technical_count = sum(1 for kw in self.technical_keywords if kw in query)

        if technical_count >= 3:
            return "high"
        elif technical_count >= 1:
            return "medium"
        else:
            return "low"

    def _estimate_chunks_needed(self, complexity: ComplexityLevel,
                                topic_count: int) -> int:
        """Estimate number of chunks needed for good answer."""
        base = {
            ComplexityLevel.SIMPLE: 3,
            ComplexityLevel.MODERATE: 5,
            ComplexityLevel.COMPLEX: 10,
            ComplexityLevel.VERY_COMPLEX: 15
        }

        return base.get(complexity, 5) + (topic_count * 2)

    def to_dict(self, classification: QueryClassification) -> Dict:
        """Convert QueryClassification to dictionary."""
        return asdict(classification)


def main():
    """Test query classifier with sample queries."""
    print("\n" + "="*60)
    print("🏷️  QUERY CLASSIFIER TEST")
    print("="*60 + "\n")

    classifier = QueryClassifier()

    # Test queries
    test_queries = [
        "What is MS Teams integration?",
        "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?",
        "Compare Slack integration vs MS Teams integration",
        "Getting error 'Authentication failed' when connecting to API"
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        result = classifier.classify(query)

        print(f"Type: {result.query_type.value}")
        print(f"Complexity: {result.complexity.value}")
        print(f"Topic Count: {result.topic_count}")
        print(f"Primary Topics: {', '.join(result.primary_topics)}")
        print(f"Expected Format: {result.expected_format.value}")
        print(f"Technical Depth: {result.technical_depth}")
        print(f"Estimated Chunks: {result.estimated_chunks_needed}")
        print(f"Requires: Images={result.requires_images}, "
              f"Code={result.requires_code}, Tables={result.requires_tables}")
        print(f"Confidence: {result.confidence:.2f}")

        print("\n" + "="*60)


if __name__ == "__main__":
    main()
