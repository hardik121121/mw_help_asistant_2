"""
Response Validator for Quality Assurance.
Validates generated answers for completeness, accuracy, and formatting.
"""

import logging
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from src.generation.answer_generator import GeneratedAnswer
from src.query.query_understanding import QueryUnderstanding
from src.retrieval.context_organizer import OrganizedContext

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of answer validation.

    Attributes:
        is_valid: Overall validation pass/fail
        completeness_score: How well all sub-questions are addressed (0-1)
        formatting_score: Quality of formatting (0-1)
        citation_score: Citation quality (0-1)
        overall_score: Weighted average score (0-1)
        issues: List of identified issues
        warnings: List of warnings
        recommendations: List of improvement suggestions
    """
    is_valid: bool
    completeness_score: float
    formatting_score: float
    citation_score: float
    overall_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ResponseValidator:
    """
    Validates generated answers for quality.

    Checks:
    - Completeness (all sub-questions addressed)
    - Formatting (structure, headings, lists)
    - Citations (proper source references)
    - Length (not too short or too long)
    - Coherence (basic quality checks)
    """

    def __init__(self,
                 min_length: int = 100,
                 max_length: int = 3000,
                 require_citations: bool = True):
        """
        Initialize response validator.

        Args:
            min_length: Minimum acceptable answer length (words)
            max_length: Maximum acceptable answer length (words)
            require_citations: Whether citations are required
        """
        self.min_length = min_length
        self.max_length = max_length
        self.require_citations = require_citations

        logger.info("Initialized ResponseValidator")

    def validate(self,
                answer: GeneratedAnswer,
                query_understanding: QueryUnderstanding,
                context: OrganizedContext) -> ValidationResult:
        """
        Validate a generated answer.

        Args:
            answer: Generated answer to validate
            query_understanding: Query analysis
            context: Retrieved context used for generation

        Returns:
            ValidationResult with scores and issues
        """
        logger.info(f"Validating answer for query: '{answer.query[:50]}...'")

        issues = []
        warnings = []
        recommendations = []

        # 1. Check completeness
        completeness_score, comp_issues = self._check_completeness(
            answer, query_understanding
        )
        issues.extend(comp_issues)

        # 2. Check formatting
        formatting_score, fmt_warnings = self._check_formatting(answer)
        warnings.extend(fmt_warnings)

        # 3. Check citations
        citation_score, cit_issues = self._check_citations(answer, context)
        if self.require_citations:
            issues.extend(cit_issues)
        else:
            warnings.extend(cit_issues)

        # 4. Check length
        length_ok, length_issues = self._check_length(answer)
        if not length_ok:
            issues.extend(length_issues)

        # 5. Check basic quality
        quality_ok, quality_warnings = self._check_basic_quality(answer)
        warnings.extend(quality_warnings)

        # Calculate overall score
        weights = {
            'completeness': 0.5,
            'formatting': 0.2,
            'citation': 0.3
        }

        overall_score = (
            weights['completeness'] * completeness_score +
            weights['formatting'] * formatting_score +
            weights['citation'] * citation_score
        )

        # Determine if valid
        is_valid = (
            len(issues) == 0 and
            overall_score >= 0.7 and
            completeness_score >= 0.6
        )

        # Generate recommendations
        if completeness_score < 0.8:
            recommendations.append("Address all sub-questions more thoroughly")
        if formatting_score < 0.7:
            recommendations.append("Improve answer structure with headings and lists")
        if citation_score < 0.5:
            recommendations.append("Add more specific citations to sources")

        result = ValidationResult(
            is_valid=is_valid,
            completeness_score=completeness_score,
            formatting_score=formatting_score,
            citation_score=citation_score,
            overall_score=overall_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )

        logger.info(f"  Validation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        logger.info(f"  Overall score: {overall_score:.2f}")
        logger.info(f"  Completeness: {completeness_score:.2f}")

        return result

    def _check_completeness(self,
                           answer: GeneratedAnswer,
                           understanding: QueryUnderstanding) -> Tuple[float, List[str]]:
        """
        Check if answer addresses all sub-questions.

        Args:
            answer: Generated answer
            understanding: Query analysis

        Returns:
            (score, issues) tuple
        """
        issues = []
        sub_questions = understanding.decomposition.sub_questions

        if not sub_questions or len(sub_questions) == 0:
            # Simple query, no sub-questions
            return 1.0, []

        answer_text = answer.answer.lower()
        addressed_count = 0

        for sq in sub_questions:
            # Extract key terms from sub-question
            question_lower = sq.question.lower()
            key_terms = self._extract_key_terms(question_lower)

            # Check if answer contains relevant information
            term_matches = sum(1 for term in key_terms if term in answer_text)
            coverage = term_matches / len(key_terms) if key_terms else 0

            if coverage >= 0.4:  # At least 40% of key terms present
                addressed_count += 1
            else:
                issues.append(f"May not fully address: '{sq.question}'")

        completeness_score = addressed_count / len(sub_questions)

        if completeness_score < 0.5:
            issues.append(f"Only {addressed_count}/{len(sub_questions)} sub-questions addressed")

        return completeness_score, issues

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from a question."""
        # Remove common question words
        stop_words = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'can', 'could', 'should', 'would',
            'is', 'are', 'the', 'a', 'an', 'to', 'and', 'or', 'in', 'on', 'for'
        }

        words = question.split()
        key_terms = [
            word.strip('?,.')
            for word in words
            if word.strip('?,.').lower() not in stop_words and len(word) > 3
        ]

        return key_terms

    def _check_formatting(self, answer: GeneratedAnswer) -> Tuple[float, List[str]]:
        """
        Check answer formatting quality.

        Args:
            answer: Generated answer

        Returns:
            (score, warnings) tuple
        """
        warnings = []
        score = 1.0

        text = answer.answer

        # Check for headings (## or **bold**)
        has_headings = bool(re.search(r'#+\s+\w+|^\*\*.+\*\*$', text, re.MULTILINE))
        if not has_headings:
            score -= 0.2
            warnings.append("Consider adding headings for better structure")

        # Check for lists
        has_lists = bool(re.search(r'^\s*[-*•]\s+\w+|^\s*\d+\.\s+\w+', text, re.MULTILINE))
        if not has_lists:
            score -= 0.2
            warnings.append("Consider using lists for clarity")

        # Check for paragraphs (not one giant block)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            score -= 0.1
            warnings.append("Break into multiple paragraphs for readability")

        # Check for very long paragraphs
        for para in paragraphs:
            if len(para.split()) > 150:
                score -= 0.1
                warnings.append("Some paragraphs are very long, consider breaking them up")
                break

        return max(score, 0.0), warnings

    def _check_citations(self,
                        answer: GeneratedAnswer,
                        context: OrganizedContext) -> Tuple[float, List[str]]:
        """
        Check citation quality.

        Args:
            answer: Generated answer
            context: Source context

        Returns:
            (score, issues) tuple
        """
        issues = []

        # Check if citations are present
        if not answer.citations or len(answer.citations) == 0:
            issues.append("No citations included")
            return 0.0, issues

        # Check if answer mentions pages or sections
        text = answer.answer
        has_page_refs = bool(re.search(r'page\s+\d+', text, re.IGNORECASE))
        has_section_refs = bool(re.search(r'section|chapter|part', text, re.IGNORECASE))

        score = 0.5  # Base score for having citations

        if has_page_refs:
            score += 0.25
        else:
            issues.append("Consider adding page number references")

        if has_section_refs:
            score += 0.25

        # Check citation coverage
        citation_coverage = len(answer.citations) / max(context.total_chunks, 1)
        if citation_coverage < 0.3:
            issues.append("Low citation coverage relative to context used")

        return min(score, 1.0), issues

    def _check_length(self, answer: GeneratedAnswer) -> Tuple[bool, List[str]]:
        """
        Check if answer length is appropriate.

        Args:
            answer: Generated answer

        Returns:
            (is_ok, issues) tuple
        """
        issues = []
        word_count = len(answer.answer.split())

        if word_count < self.min_length:
            issues.append(f"Answer too short ({word_count} words, min: {self.min_length})")
            return False, issues

        if word_count > self.max_length:
            issues.append(f"Answer too long ({word_count} words, max: {self.max_length})")
            return False, issues

        return True, []

    def _check_basic_quality(self, answer: GeneratedAnswer) -> Tuple[bool, List[str]]:
        """
        Basic quality checks.

        Args:
            answer: Generated answer

        Returns:
            (is_ok, warnings) tuple
        """
        warnings = []
        text = answer.answer

        # Check for obvious errors
        if "i don't know" in text.lower() or "cannot answer" in text.lower():
            warnings.append("Answer indicates uncertainty or inability to respond")

        # Check for repetition (same sentence appearing multiple times)
        sentences = re.split(r'[.!?]+', text)
        sentence_counts = {}
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if len(sent_clean) > 20:  # Only check substantial sentences
                sentence_counts[sent_clean] = sentence_counts.get(sent_clean, 0) + 1

        for sent, count in sentence_counts.items():
            if count > 1:
                warnings.append("Detected repeated sentences")
                break

        # Check for incomplete sentences at the end
        if text.strip() and not text.strip()[-1] in '.!?':
            warnings.append("Answer may be incomplete (doesn't end with punctuation)")

        return len(warnings) == 0, warnings


if __name__ == "__main__":
    """Test response validator."""
    print("\n" + "="*70)
    print("TESTING RESPONSE VALIDATOR")
    print("="*70 + "\n")

    validator = ResponseValidator()
    print(f"✅ ResponseValidator initialized")
    print(f"   Min length: {validator.min_length} words")
    print(f"   Max length: {validator.max_length} words")
    print(f"   Require citations: {validator.require_citations}")

    print("\n✅ Response validator ready!")
    print("\nFor full testing, use Phase 6 test script with generated answers.\n")
