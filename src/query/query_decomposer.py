"""
Query Decomposition System for Complex Multi-Topic Queries.
Breaks down complex queries into manageable sub-questions using LLM.
"""

import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from groq import Groq
except ImportError:
    print("⚠️  Groq not installed. Please run: pip install groq")
    Groq = None

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DependencyType(str, Enum):
    """Types of dependencies between sub-questions."""
    INDEPENDENT = "independent"  # Can be answered in parallel
    SEQUENTIAL = "sequential"    # Must be answered in order
    CONDITIONAL = "conditional"  # Answer depends on previous results


@dataclass
class SubQuestion:
    """Represents a decomposed sub-question."""
    id: str
    question: str
    topics: List[str]
    dependency_type: DependencyType = DependencyType.INDEPENDENT
    depends_on: List[str] = field(default_factory=list)  # IDs of questions this depends on
    priority: int = 1  # 1=highest priority
    reasoning: Optional[str] = None  # Why this sub-question is needed


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original_query: str
    sub_questions: List[SubQuestion]
    query_complexity: str  # "simple", "moderate", "complex", "very_complex"
    should_decompose: bool
    total_topics: int
    reasoning: Optional[str] = None


class QueryDecomposer:
    """
    Decomposes complex queries into sub-questions for multi-step retrieval.

    Uses LLM to analyze the query and break it down into logical components
    that can be answered independently or sequentially.
    """

    def __init__(self):
        """Initialize query decomposer with Groq LLM."""
        self.settings = get_settings()

        if Groq is None:
            raise RuntimeError("Groq library not installed")

        self.client = Groq(api_key=self.settings.groq_api_key)
        self.model = self.settings.llm_model
        self.temperature = 0.2  # Lower for more deterministic decomposition

        logger.info(f"Initialized QueryDecomposer with model: {self.model}")

    def decompose(self, query: str) -> DecomposedQuery:
        """
        Decompose a query into sub-questions if needed.

        Args:
            query: User's query string

        Returns:
            DecomposedQuery with sub-questions and metadata
        """
        logger.info(f"Decomposing query: {query}")

        # Check if decomposition is needed
        if not self._should_decompose(query):
            logger.info("Query is simple, no decomposition needed")
            return DecomposedQuery(
                original_query=query,
                sub_questions=[SubQuestion(
                    id="q1",
                    question=query,
                    topics=self._extract_topics(query),
                    priority=1
                )],
                query_complexity="simple",
                should_decompose=False,
                total_topics=1,
                reasoning="Query is simple and focused on a single topic"
            )

        # Use LLM to decompose complex query
        decomposition = self._llm_decompose(query)

        logger.info(f"Decomposed into {len(decomposition.sub_questions)} sub-questions")
        return decomposition

    def _should_decompose(self, query: str) -> bool:
        """
        Determine if a query needs decomposition.

        Checks:
        - Word count threshold
        - Presence of multiple topics/questions
        - Conjunction words (and, or, then, also)
        """
        words = query.split()

        # Short queries typically don't need decomposition
        if len(words) < self.settings.query_complexity_threshold:
            return False

        # Check for multiple question indicators
        indicators = ["and", "then", "also", "additionally", "furthermore",
                     "how do i", "what is", "explain", "steps to"]

        indicator_count = sum(1 for word in indicators if word in query.lower())

        # Multiple indicators suggest complex query
        return indicator_count >= 2

    def _llm_decompose(self, query: str) -> DecomposedQuery:
        """
        Use LLM to decompose query into sub-questions.

        Args:
            query: User's query

        Returns:
            DecomposedQuery with structured sub-questions
        """
        prompt = self._build_decomposition_prompt(query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing and decomposing complex documentation queries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=2000
            )

            result = response.choices[0].message.content
            logger.debug(f"LLM decomposition result: {result}")

            # Parse LLM response into structured format
            return self._parse_llm_response(query, result)

        except Exception as e:
            logger.error(f"Error in LLM decomposition: {e}")
            # Fallback to simple decomposition
            return self._fallback_decomposition(query)

    def _build_decomposition_prompt(self, query: str) -> str:
        """Build prompt for LLM decomposition with domain-specific guidance."""
        return f"""You are analyzing a query about Watermelon (a test automation platform).
Break down complex queries into sub-questions that enable effective documentation retrieval.

Query: "{query}"

DOMAIN CONTEXT:
- Watermelon is a test automation platform with features like:
  * No-code test creation and automation
  * Autonomous Functional Testing (AFT)
  * Integrations (MS Teams, Slack, Shopify, Jira, etc.)
  * API testing and parameterization
  * Visual AI for test recording
  * Device farms and grid testing

DECOMPOSITION GUIDELINES:

1. WHEN TO DECOMPOSE:
   - Multiple distinct topics/features mentioned (e.g., "X and Y")
   - Multi-step procedures (e.g., "create X then process for Y")
   - Conceptual + procedural mix (e.g., "what is X and how to use it")
   - Integration + configuration (e.g., "integrate X and configure Y")

2. WHEN NOT TO DECOMPOSE:
   - Single focused question about one feature
   - Simple "what is" or "how to" questions
   - Questions about a single integration or feature

3. HOW TO CREATE EFFECTIVE SUB-QUESTIONS:
   - Make each sub-question SELF-CONTAINED and SPECIFIC
   - Use exact feature/integration names (e.g., "MS Teams", not just "integration")
   - For procedural queries, separate "what" from "how"
   - Maintain context from original query in each sub-question
   - Keep sub-questions focused (one topic per question)
   - Use the SAME terminology as the original query

4. EXAMPLES OF GOOD DECOMPOSITION:

   Query: "How do I integrate MS Teams and configure automated notifications?"
   → Sub-Q1: "How do I integrate MS Teams with Watermelon?"
   → Sub-Q2: "How do I configure automated notifications in MS Teams integration?"

   Query: "What is Autonomous Functional Testing and how do I use it with no-code blocks?"
   → Sub-Q1: "What is Autonomous Functional Testing in Watermelon?"
   → Sub-Q2: "How do I use Autonomous Functional Testing with no-code blocks?"

   Query: "How do I set up API testing?"  [SIMPLE - NO DECOMPOSITION]
   → Sub-Q1: Original query only

5. DEPENDENCY TYPES:
   - "independent": Can answer in parallel (e.g., two unrelated features)
   - "sequential": Must answer in order (e.g., "what is X" before "how to use X")
   - "conditional": Answer depends on previous results (rarely used)

Respond ONLY with valid JSON in this exact format:
{{
    "query_complexity": "simple|moderate|complex|very_complex",
    "should_decompose": true|false,
    "reasoning": "Brief explanation (1 sentence)",
    "sub_questions": [
        {{
            "id": "q1",
            "question": "Self-contained specific sub-question",
            "topics": ["specific_feature", "integration_name"],
            "dependency_type": "independent|sequential|conditional",
            "depends_on": [],
            "priority": 1,
            "reasoning": "Why this sub-question helps retrieval (1 sentence)"
        }}
    ]
}}

IMPORTANT:
- If simple query, set should_decompose=false and include ONE sub-question with original query
- Keep sub-questions specific and self-contained
- Use exact names/terms from original query
- Ensure valid JSON syntax (no trailing commas, proper escaping)"""

    def _parse_llm_response(self, original_query: str, llm_response: str) -> DecomposedQuery:
        """
        Parse LLM JSON response into DecomposedQuery object.

        Args:
            original_query: Original user query
            llm_response: JSON response from LLM

        Returns:
            Parsed DecomposedQuery
        """
        try:
            # Extract JSON from response (sometimes LLM adds markdown)
            json_str = llm_response.strip()
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0]
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            # Convert to SubQuestion objects
            sub_questions = []
            for sq in data.get("sub_questions", []):
                sub_questions.append(SubQuestion(
                    id=sq["id"],
                    question=sq["question"],
                    topics=sq.get("topics", []),
                    dependency_type=DependencyType(sq.get("dependency_type", "independent")),
                    depends_on=sq.get("depends_on", []),
                    priority=sq.get("priority", 1),
                    reasoning=sq.get("reasoning")
                ))

            return DecomposedQuery(
                original_query=original_query,
                sub_questions=sub_questions,
                query_complexity=data.get("query_complexity", "moderate"),
                should_decompose=data.get("should_decompose", True),
                total_topics=len(set(topic for sq in sub_questions for topic in sq.topics)),
                reasoning=data.get("reasoning")
            )

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response was: {llm_response}")
            return self._fallback_decomposition(original_query)

    def _fallback_decomposition(self, query: str) -> DecomposedQuery:
        """
        Fallback decomposition when LLM fails.

        Uses simple heuristics to break down query.
        """
        logger.warning("Using fallback decomposition")

        # Simple rule-based decomposition
        sub_questions = []

        # Split on common conjunctions
        parts = []
        for separator in [" and ", " then ", " also "]:
            if separator in query.lower():
                parts = query.split(separator, 1)
                break

        if not parts:
            parts = [query]

        for i, part in enumerate(parts[:4], 1):  # Max 4 sub-questions
            sub_questions.append(SubQuestion(
                id=f"q{i}",
                question=part.strip(),
                topics=self._extract_topics(part),
                priority=i,
                dependency_type=DependencyType.SEQUENTIAL if i > 1 else DependencyType.INDEPENDENT,
                depends_on=[f"q{i-1}"] if i > 1 else []
            ))

        return DecomposedQuery(
            original_query=query,
            sub_questions=sub_questions,
            query_complexity="moderate",
            should_decompose=len(sub_questions) > 1,
            total_topics=len(sub_questions),
            reasoning="Fallback decomposition using rule-based splitting"
        )

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract likely topics from text using simple keyword extraction.

        This is a basic implementation; could be enhanced with NER or embeddings.
        """
        # Simple topic extraction - look for capitalized phrases and key terms
        words = text.split()
        topics = []

        # Multi-word capitalized phrases
        current_phrase = []
        for word in words:
            clean_word = word.strip(",.?!:;")
            if clean_word and clean_word[0].isupper():
                current_phrase.append(clean_word)
            else:
                if len(current_phrase) >= 2:
                    topics.append(" ".join(current_phrase))
                current_phrase = []

        if len(current_phrase) >= 2:
            topics.append(" ".join(current_phrase))

        # Key technical terms
        technical_terms = ["api", "integration", "authentication", "testing",
                          "deployment", "configuration", "setup", "workflow"]

        text_lower = text.lower()
        for term in technical_terms:
            if term in text_lower:
                topics.append(term)

        return list(set(topics)) if topics else ["general"]

    def to_dict(self, decomposed: DecomposedQuery) -> Dict:
        """Convert DecomposedQuery to dictionary for serialization."""
        return asdict(decomposed)

    def from_dict(self, data: Dict) -> DecomposedQuery:
        """Load DecomposedQuery from dictionary."""
        sub_questions = [
            SubQuestion(**sq) for sq in data.get("sub_questions", [])
        ]

        return DecomposedQuery(
            original_query=data["original_query"],
            sub_questions=sub_questions,
            query_complexity=data.get("query_complexity", "moderate"),
            should_decompose=data.get("should_decompose", True),
            total_topics=data.get("total_topics", 1),
            reasoning=data.get("reasoning")
        )


def main():
    """Test query decomposition with sample queries."""
    print("\n" + "="*60)
    print("🔍 QUERY DECOMPOSER TEST")
    print("="*60 + "\n")

    decomposer = QueryDecomposer()

    # Test queries
    test_queries = [
        "What is MS Teams integration?",
        "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?",
        "What are the integration steps for MS Teams and how do I configure automated responses?",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)

        result = decomposer.decompose(query)

        print(f"Complexity: {result.query_complexity}")
        print(f"Should Decompose: {result.should_decompose}")
        print(f"Total Topics: {result.total_topics}")

        if result.reasoning:
            print(f"Reasoning: {result.reasoning}")

        print(f"\nSub-questions ({len(result.sub_questions)}):")
        for sq in result.sub_questions:
            print(f"\n  [{sq.id}] {sq.question}")
            print(f"      Topics: {', '.join(sq.topics)}")
            print(f"      Dependency: {sq.dependency_type.value}")
            if sq.depends_on:
                print(f"      Depends on: {', '.join(sq.depends_on)}")
            print(f"      Priority: {sq.priority}")

        print("\n" + "="*60)


if __name__ == "__main__":
    main()
