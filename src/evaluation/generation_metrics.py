"""
Generation Metrics Calculator.
Evaluates answer quality using various metrics.
"""

import logging
import re
from typing import List, Dict, Set
from dataclasses import dataclass, field
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """
    Answer generation evaluation metrics.

    Attributes:
        completeness_score: How well all sub-questions addressed (0-1)
        coherence_score: Text coherence and flow (0-1)
        formatting_score: Structure and formatting quality (0-1)
        citation_score: Citation quality (0-1)
        length_score: Appropriate length (0-1)
        keyword_coverage: Expected keyword coverage (0-1)
        overall_score: Weighted average (0-1)
        word_count: Total words in answer
        sentence_count: Total sentences
        avg_sentence_length: Average words per sentence
        has_headings: Whether answer has section headings
        has_lists: Whether answer has lists
        has_citations: Whether answer has citations
    """
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    formatting_score: float = 0.0
    citation_score: float = 0.0
    length_score: float = 0.0
    keyword_coverage: float = 0.0
    overall_score: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    has_headings: bool = False
    has_lists: bool = False
    has_citations: bool = False


class GenerationMetricsCalculator:
    """
    Calculate answer quality metrics.

    Evaluates:
    - Completeness (sub-question coverage)
    - Coherence (readability, flow)
    - Formatting (structure, organization)
    - Citations (source references)
    - Length (appropriate for query)
    - Keyword coverage (expected terms present)
    """

    def __init__(self,
                 min_length: int = 100,
                 max_length: int = 1500,
                 ideal_length: int = 500):
        """
        Initialize metrics calculator.

        Args:
            min_length: Minimum acceptable word count
            max_length: Maximum acceptable word count
            ideal_length: Ideal word count
        """
        self.min_length = min_length
        self.max_length = max_length
        self.ideal_length = ideal_length

        logger.info("Initialized GenerationMetricsCalculator")

    def calculate_metrics(self,
                         answer_text: str,
                         expected_topics: List[str],
                         sub_questions: List[str] = None,
                         has_citations: bool = False) -> GenerationMetrics:
        """
        Calculate all generation metrics.

        Args:
            answer_text: Generated answer text
            expected_topics: Expected topic keywords
            sub_questions: List of sub-questions to address
            has_citations: Whether answer includes citations

        Returns:
            GenerationMetrics with all scores
        """
        logger.info("Calculating generation metrics...")

        metrics = GenerationMetrics()

        # Basic statistics
        metrics.word_count = len(answer_text.split())
        sentences = self._split_sentences(answer_text)
        metrics.sentence_count = len(sentences)
        metrics.avg_sentence_length = (
            metrics.word_count / metrics.sentence_count
            if metrics.sentence_count > 0 else 0
        )

        # Formatting checks
        metrics.has_headings = self._has_headings(answer_text)
        metrics.has_lists = self._has_lists(answer_text)
        metrics.has_citations = has_citations

        # Score calculations
        metrics.completeness_score = self._calculate_completeness(
            answer_text, sub_questions or []
        )
        metrics.coherence_score = self._calculate_coherence(
            answer_text, sentences
        )
        metrics.formatting_score = self._calculate_formatting_score(
            answer_text, metrics
        )
        metrics.citation_score = 1.0 if has_citations else 0.0
        metrics.length_score = self._calculate_length_score(metrics.word_count)
        metrics.keyword_coverage = self._calculate_keyword_coverage(
            answer_text, expected_topics
        )

        # Overall score (weighted average)
        weights = {
            'completeness': 0.30,
            'coherence': 0.15,
            'formatting': 0.15,
            'citation': 0.15,
            'length': 0.10,
            'keywords': 0.15
        }

        metrics.overall_score = (
            weights['completeness'] * metrics.completeness_score +
            weights['coherence'] * metrics.coherence_score +
            weights['formatting'] * metrics.formatting_score +
            weights['citation'] * metrics.citation_score +
            weights['length'] * metrics.length_score +
            weights['keywords'] * metrics.keyword_coverage
        )

        logger.info(f"  Overall score: {metrics.overall_score:.3f}")
        logger.info(f"  Completeness: {metrics.completeness_score:.3f}")
        logger.info(f"  Word count: {metrics.word_count}")

        return metrics

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _has_headings(self, text: str) -> bool:
        """Check if text has section headings."""
        # Look for markdown headings or bold text
        return bool(re.search(r'#+\s+\w+|^\*\*.+\*\*$', text, re.MULTILINE))

    def _has_lists(self, text: str) -> bool:
        """Check if text has lists."""
        return bool(re.search(r'^\s*[-*•]\s+\w+|^\s*\d+\.\s+\w+', text, re.MULTILINE))

    def _calculate_completeness(self,
                                answer: str,
                                sub_questions: List[str]) -> float:
        """
        Calculate completeness score.

        Checks if answer addresses all sub-questions.
        """
        if not sub_questions:
            return 1.0  # No sub-questions to check

        answer_lower = answer.lower()
        addressed = 0

        for sq in sub_questions:
            # Extract key terms from sub-question
            key_terms = self._extract_key_terms(sq)

            # Check if sufficient terms are in answer
            found_terms = sum(1 for term in key_terms if term in answer_lower)
            coverage = found_terms / len(key_terms) if key_terms else 0

            if coverage >= 0.4:  # At least 40% of key terms
                addressed += 1

        return addressed / len(sub_questions)

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        stop_words = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'can', 'could', 'should', 'would',
            'is', 'are', 'the', 'a', 'an', 'to', 'and', 'or', 'in', 'on', 'for'
        }

        words = text.lower().split()
        key_terms = [
            word.strip('?,.')
            for word in words
            if word.strip('?,.') not in stop_words and len(word) > 3
        ]

        return key_terms

    def _calculate_coherence(self,
                            answer: str,
                            sentences: List[str]) -> float:
        """
        Calculate coherence score.

        Based on:
        - Sentence variety
        - No excessive repetition
        - Reasonable sentence length
        """
        if not sentences:
            return 0.0

        score = 1.0

        # Check for sentence variety
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            avg_len = sum(sentence_lengths) / len(sentence_lengths)
            std_dev = (sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)) ** 0.5

            # Penalize if too uniform or too varied
            if std_dev < 2:
                score -= 0.1  # Too uniform
            elif std_dev > 15:
                score -= 0.1  # Too varied

        # Check for repetition
        word_freq = Counter(answer.lower().split())
        max_freq = max(word_freq.values()) if word_freq else 0
        if max_freq > len(sentences) * 2:  # Same word appearing too often
            score -= 0.2

        # Check sentence length appropriateness
        for sent_len in sentence_lengths:
            if sent_len > 50:  # Very long sentence
                score -= 0.05
            elif sent_len < 3:  # Very short sentence
                score -= 0.05

        return max(score, 0.0)

    def _calculate_formatting_score(self,
                                   answer: str,
                                   metrics: GenerationMetrics) -> float:
        """Calculate formatting quality score."""
        score = 0.0

        # Has headings?
        if metrics.has_headings:
            score += 0.35

        # Has lists?
        if metrics.has_lists:
            score += 0.35

        # Has paragraphs?
        paragraphs = [p.strip() for p in answer.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.30

        return min(score, 1.0)

    def _calculate_length_score(self, word_count: int) -> float:
        """Calculate length appropriateness score."""
        if word_count < self.min_length:
            # Too short
            return word_count / self.min_length

        if word_count > self.max_length:
            # Too long
            excess = word_count - self.max_length
            penalty = min(excess / self.max_length, 0.5)
            return 1.0 - penalty

        # In acceptable range, score based on closeness to ideal
        if word_count <= self.ideal_length:
            # Below ideal, score increases toward ideal
            return 0.7 + 0.3 * (word_count / self.ideal_length)
        else:
            # Above ideal, score decreases
            excess = word_count - self.ideal_length
            max_excess = self.max_length - self.ideal_length
            penalty = 0.3 * (excess / max_excess)
            return 1.0 - penalty

    def _calculate_keyword_coverage(self,
                                   answer: str,
                                   expected_topics: List[str]) -> float:
        """Calculate coverage of expected keywords/topics."""
        if not expected_topics:
            return 1.0

        answer_lower = answer.lower()
        covered = 0

        for topic in expected_topics:
            topic_keywords = topic.lower().split()
            # Topic is covered if any of its keywords appear
            if any(kw in answer_lower for kw in topic_keywords):
                covered += 1

        return covered / len(expected_topics)


if __name__ == "__main__":
    """Test generation metrics calculator."""
    print("\n" + "="*70)
    print("TESTING GENERATION METRICS CALCULATOR")
    print("="*70 + "\n")

    calculator = GenerationMetricsCalculator()
    print(f"✅ GenerationMetricsCalculator initialized")
    print(f"   Ideal length: {calculator.ideal_length} words")

    # Sample answer
    test_answer = """
## How to Create No-Code Blocks

To create a no-code block on the Watermelon platform, follow these steps:

1. Navigate to the Blocks section in your dashboard
2. Click "Create New Block"
3. Select "No-Code Block" from the options
4. Configure your block settings

The process is straightforward and requires no programming knowledge.

## Testing Setup

For Autonomous Functional Testing, you'll need to:

- Connect your block to the testing framework
- Configure test parameters
- Run initial validation tests

This ensures your block works correctly before deployment.
    """.strip()

    metrics = calculator.calculate_metrics(
        answer_text=test_answer,
        expected_topics=['no-code blocks', 'testing', 'watermelon'],
        sub_questions=[
            'How to create no-code blocks?',
            'What is testing setup?'
        ],
        has_citations=True
    )

    print(f"\n📊 Sample Metrics:")
    print(f"  Overall Score: {metrics.overall_score:.3f}")
    print(f"  Completeness: {metrics.completeness_score:.3f}")
    print(f"  Coherence: {metrics.coherence_score:.3f}")
    print(f"  Formatting: {metrics.formatting_score:.3f}")
    print(f"  Citation: {metrics.citation_score:.3f}")
    print(f"  Length: {metrics.length_score:.3f}")
    print(f"  Keyword Coverage: {metrics.keyword_coverage:.3f}")
    print(f"\n  Word Count: {metrics.word_count}")
    print(f"  Has Headings: {metrics.has_headings}")
    print(f"  Has Lists: {metrics.has_lists}")

    print("\n✅ Generation metrics calculator ready!\n")
