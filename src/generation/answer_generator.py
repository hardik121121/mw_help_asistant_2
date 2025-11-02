"""
Answer Generator for Multi-Context RAG.
Generates comprehensive answers using LLM with organized context.
"""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from groq import Groq
except ImportError:
    print("⚠️  Groq not installed. Please run: pip install groq")
    Groq = None

from config.settings import get_settings
from src.retrieval.context_organizer import OrganizedContext
from src.query.query_understanding import QueryUnderstanding

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """
    Generated answer with metadata.

    Attributes:
        answer: The generated answer text
        query: Original query
        sub_answers: Answers to individual sub-questions
        citations: Source references
        images_used: List of image paths referenced
        generation_time: Time taken (seconds)
        model: LLM model used
        tokens_used: Approximate token count
        confidence: Quality confidence score (0-1)
    """
    answer: str
    query: str
    sub_answers: List[Dict] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    images_used: List[str] = field(default_factory=list)
    generation_time: float = 0.0
    model: str = ""
    tokens_used: int = 0
    confidence: float = 0.0


class AnswerGenerator:
    """
    Generates comprehensive answers using LLM with retrieved context.

    Features:
    - Multi-context prompt engineering
    - Strategy-aware generation (step-by-step, comparison, etc.)
    - Citation integration
    - Image reference
    - Response formatting
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialize answer generator.

        Args:
            model: Groq model name (default from settings)
        """
        self.settings = get_settings()

        if Groq is None:
            raise RuntimeError("Groq library not installed")

        self.client = Groq(api_key=self.settings.groq_api_key)
        self.model = model or self.settings.llm_model
        self.temperature = self.settings.llm_temperature
        self.max_tokens = self.settings.llm_max_tokens

        logger.info(f"Initialized AnswerGenerator with model: {self.model}")

    def generate(self,
                query: str,
                context: OrganizedContext,
                query_understanding: QueryUnderstanding) -> GeneratedAnswer:
        """
        Generate comprehensive answer for a query.

        Args:
            query: Original user query
            context: Organized retrieval context
            query_understanding: Query analysis from Phase 3

        Returns:
            GeneratedAnswer with formatted response
        """
        logger.info(f"Generating answer for: '{query[:50]}...'")
        start_time = time.time()

        # Build prompt based on generation strategy
        strategy = query_understanding.generation_strategy
        logger.info(f"  Strategy: {strategy}")

        if strategy == "step_by_step":
            prompt = self._build_step_by_step_prompt(query, context, query_understanding)
        elif strategy == "comparison":
            prompt = self._build_comparison_prompt(query, context, query_understanding)
        elif strategy == "troubleshooting":
            prompt = self._build_troubleshooting_prompt(query, context, query_understanding)
        else:
            prompt = self._build_standard_prompt(query, context, query_understanding)

        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(strategy)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            answer_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            elapsed = time.time() - start_time

            logger.info(f"✅ Answer generated in {elapsed:.2f}s")
            logger.info(f"   Tokens: {tokens_used}")

            # Extract citations and images
            citations = self._extract_citations(context)
            images = self._extract_images(context)

            result = GeneratedAnswer(
                answer=answer_text,
                query=query,
                citations=citations,
                images_used=images,
                generation_time=elapsed,
                model=self.model,
                tokens_used=tokens_used,
                confidence=self._estimate_confidence(context, answer_text)
            )

            return result

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _get_system_prompt(self, strategy: str) -> str:
        """Get system prompt based on generation strategy."""
        base = """You are a helpful AI assistant for the Watermelon customer service platform.
Your role is to provide accurate, comprehensive answers based on the official documentation provided."""

        if strategy == "step_by_step":
            return base + """

When answering:
1. Provide clear, numbered step-by-step instructions
2. Include all necessary prerequisites
3. Explain what each step accomplishes
4. Mention relevant screenshots or visual aids when available
5. Include tips or warnings where appropriate"""

        elif strategy == "comparison":
            return base + """

When answering:
1. Create a clear comparison structure
2. List similarities and differences
3. Provide recommendations based on use cases
4. Be objective and factual"""

        elif strategy == "troubleshooting":
            return base + """

When answering:
1. Identify the likely problem
2. Provide diagnostic steps
3. Suggest solutions in order of likelihood
4. Include preventive measures
5. Mention when to contact support"""

        else:  # standard
            return base + """

When answering:
1. Be comprehensive yet concise
2. Structure information logically
3. Use headings and bullet points for clarity
4. Cite specific sections when helpful
5. Include relevant examples"""

    def _build_standard_prompt(self,
                              query: str,
                              context: OrganizedContext,
                              understanding: QueryUnderstanding) -> str:
        """Build standard generation prompt."""
        context_text = self._format_context(context)

        prompt = f"""Question: {query}

Based on the following documentation from Watermelon, provide a comprehensive answer:

{context_text}

Instructions:
- Answer the question thoroughly
- Use information from all relevant sections above
- Structure your answer with clear headings
- Be specific and accurate
- Mention page numbers for key information when appropriate

Answer:"""

        return prompt

    def _build_step_by_step_prompt(self,
                                  query: str,
                                  context: OrganizedContext,
                                  understanding: QueryUnderstanding) -> str:
        """Build step-by-step generation prompt."""
        context_text = self._format_context(context)

        sub_questions = understanding.decomposition.sub_questions
        sub_q_text = "\n".join([f"  {i+1}. {sq.question}" for i, sq in enumerate(sub_questions)])

        prompt = f"""Question: {query}

This question requires a step-by-step answer covering:
{sub_q_text}

Documentation context:

{context_text}

Instructions:
- Provide clear, numbered steps
- Start with prerequisites (if any)
- Explain each step in detail
- Include expected outcomes
- Mention relevant screenshots when available (from the context above)
- Add tips or warnings where appropriate

Step-by-Step Answer:"""

        return prompt

    def _build_comparison_prompt(self,
                                query: str,
                                context: OrganizedContext,
                                understanding: QueryUnderstanding) -> str:
        """Build comparison generation prompt."""
        context_text = self._format_context(context)

        prompt = f"""Question: {query}

Documentation context:

{context_text}

Instructions:
- Compare the features/options mentioned in the question
- Structure as: Overview → Similarities → Differences → Recommendations
- Use tables or bullet points for clarity
- Be objective and factual
- Cite specific capabilities from the documentation

Comparison:"""

        return prompt

    def _build_troubleshooting_prompt(self,
                                     query: str,
                                     context: OrganizedContext,
                                     understanding: QueryUnderstanding) -> str:
        """Build troubleshooting generation prompt."""
        context_text = self._format_context(context)

        prompt = f"""Question: {query}

Documentation context:

{context_text}

Instructions:
- Identify the likely problem based on the question
- Provide diagnostic steps
- Suggest solutions (most likely first)
- Include preventive measures
- Mention when to contact support

Troubleshooting Guide:"""

        return prompt

    def _format_context(self, context: OrganizedContext) -> str:
        """Format organized context for LLM prompt."""
        sections = []

        sections.append(f"Retrieved Information ({context.total_chunks} relevant sections):")
        sections.append("")

        # Format by topic
        for topic, chunks in context.topic_groups.items():
            sections.append(f"\n## Topic: {topic}")
            sections.append("")

            for i, chunk in enumerate(chunks, 1):
                metadata = chunk.get('metadata', {})
                heading_path = metadata.get('heading_path', [])
                page_start = metadata.get('page_start', 0)
                content = chunk['content']

                # Format heading
                if heading_path:
                    heading = " > ".join(heading_path)
                else:
                    heading = "General Information"

                sections.append(f"### [{i}] {heading} (Page {page_start})")

                # Add content
                sections.append(content)
                sections.append("")

                # Note images/tables if present
                if metadata.get('has_images'):
                    image_paths = metadata.get('image_paths', [])
                    if image_paths:
                        sections.append(f"*[Images available: {', '.join(image_paths)}]*")
                        sections.append("")

                if metadata.get('has_tables'):
                    sections.append("*[Contains data tables]*")
                    sections.append("")

        return "\n".join(sections)

    def _extract_citations(self, context: OrganizedContext) -> List[str]:
        """Extract citation information from context."""
        citations = []

        for chunk in context.chunks:
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])
            page_start = metadata.get('page_start', 0)

            if heading_path:
                citation = f"{' > '.join(heading_path)} (Page {page_start})"
            else:
                citation = f"Page {page_start}"

            if citation not in citations:
                citations.append(citation)

        return citations

    def _extract_images(self, context: OrganizedContext) -> List[str]:
        """Extract relevant images from context."""
        images = []

        for chunk in context.chunks:
            metadata = chunk.get('metadata', {})
            if metadata.get('has_images'):
                image_paths = metadata.get('image_paths', [])
                images.extend(image_paths)

        # Remove duplicates while preserving order
        seen = set()
        unique_images = []
        for img in images:
            if img not in seen:
                seen.add(img)
                unique_images.append(img)

        return unique_images

    def _estimate_confidence(self, context: OrganizedContext, answer: str) -> float:
        """Estimate confidence in answer quality."""
        confidence = 0.5  # Base confidence

        # More chunks = higher confidence (up to a point)
        if context.total_chunks >= 15:
            confidence += 0.2
        elif context.total_chunks >= 10:
            confidence += 0.1

        # More sections = better coverage
        if context.unique_sections >= 5:
            confidence += 0.1
        elif context.unique_sections >= 3:
            confidence += 0.05

        # Has images/tables = richer context
        if context.has_images:
            confidence += 0.05
        if context.has_tables:
            confidence += 0.05

        # Longer answer (but not too long) = more thorough
        answer_length = len(answer.split())
        if 200 <= answer_length <= 1000:
            confidence += 0.1
        elif 100 <= answer_length < 200:
            confidence += 0.05

        return min(confidence, 1.0)


if __name__ == "__main__":
    """Test answer generator."""
    print("\n" + "="*70)
    print("TESTING ANSWER GENERATOR")
    print("="*70 + "\n")

    # This would require full pipeline integration
    # For now, just test initialization
    generator = AnswerGenerator()
    print(f"✅ AnswerGenerator initialized")
    print(f"   Model: {generator.model}")
    print(f"   Temperature: {generator.temperature}")
    print(f"   Max tokens: {generator.max_tokens}")

    print("\n✅ Answer generator ready!")
    print("\nFor full testing, use Phase 6 test script with complete pipeline.\n")
