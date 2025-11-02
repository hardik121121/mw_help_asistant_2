"""
End-to-End RAG Pipeline.
Integrates query understanding, retrieval, and generation.
"""

import logging
import time
from typing import Dict, Optional
from dataclasses import dataclass, asdict

from src.query.query_understanding import QueryUnderstandingEngine, QueryUnderstanding
from src.retrieval.multi_step_retriever import MultiStepRetriever, RetrievalResult
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer
from src.generation.response_validator import ResponseValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Complete pipeline execution result.

    Attributes:
        query: Original query
        understanding: Query analysis
        retrieval: Retrieval results
        answer: Generated answer
        validation: Validation results
        total_time: End-to-end time (seconds)
        success: Whether pipeline completed successfully
    """
    query: str
    understanding: QueryUnderstanding
    retrieval: RetrievalResult
    answer: GeneratedAnswer
    validation: ValidationResult
    total_time: float
    success: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'query': self.query,
            'understanding': {
                'type': self.understanding.classification.query_type.value,
                'complexity': self.understanding.classification.complexity.value,
                'num_sub_questions': len(self.understanding.decomposition.sub_questions),
                'strategy': self.understanding.generation_strategy
            },
            'retrieval': {
                'total_retrieved': self.retrieval.total_chunks_retrieved,
                'final_chunks': self.retrieval.final_chunks,
                'unique_sections': self.retrieval.organized_context.unique_sections,
                'time': self.retrieval.retrieval_time
            },
            'answer': {
                'text': self.answer.answer,
                'citations': self.answer.citations,
                'images': self.answer.images_used,
                'tokens': self.answer.tokens_used,
                'time': self.answer.generation_time,
                'confidence': self.answer.confidence
            },
            'validation': {
                'is_valid': self.validation.is_valid,
                'overall_score': self.validation.overall_score,
                'completeness': self.validation.completeness_score,
                'formatting': self.validation.formatting_score,
                'citation': self.validation.citation_score,
                'issues': self.validation.issues,
                'warnings': self.validation.warnings
            },
            'total_time': self.total_time,
            'success': self.success
        }


class EndToEndPipeline:
    """
    Complete RAG pipeline from query to validated answer.

    Pipeline stages:
    1. Query Understanding (Phase 3)
    2. Multi-Step Retrieval (Phase 4)
    3. Answer Generation (Phase 6)
    4. Response Validation (Phase 6)
    """

    def __init__(self,
                 use_reranking: bool = True,
                 enable_context_chaining: bool = True,
                 validate_responses: bool = True):
        """
        Initialize end-to-end pipeline.

        Args:
            use_reranking: Enable Cohere reranking in retrieval
            enable_context_chaining: Use context chaining in retrieval
            validate_responses: Validate generated answers
        """
        logger.info("="*70)
        logger.info("INITIALIZING END-TO-END RAG PIPELINE")
        logger.info("="*70)

        # Initialize all components
        logger.info("\n1. Initializing Query Understanding Engine...")
        self.query_engine = QueryUnderstandingEngine()

        logger.info("\n2. Initializing Multi-Step Retriever...")
        self.retriever = MultiStepRetriever(
            use_reranking=use_reranking,
            enable_context_chaining=enable_context_chaining
        )

        logger.info("\n3. Initializing Answer Generator...")
        self.generator = AnswerGenerator()

        if validate_responses:
            logger.info("\n4. Initializing Response Validator...")
            self.validator = ResponseValidator()
        else:
            self.validator = None

        logger.info("\n" + "="*70)
        logger.info("✅ PIPELINE READY")
        logger.info("="*70 + "\n")

    def process_query(self, query: str, max_chunks: int = 20) -> PipelineResult:
        """
        Process a query end-to-end.

        Args:
            query: User query
            max_chunks: Maximum chunks for generation

        Returns:
            PipelineResult with all intermediate and final results
        """
        logger.info("\n" + "="*70)
        logger.info(f"PROCESSING QUERY: {query}")
        logger.info("="*70 + "\n")

        start_time = time.time()
        success = True

        try:
            # Stage 1: Query Understanding
            logger.info("📋 STAGE 1: QUERY UNDERSTANDING")
            logger.info("-" * 70)
            understanding = self.query_engine.understand(query)
            logger.info(f"✅ Query understood")
            logger.info(f"   Type: {understanding.classification.query_type.value}")
            logger.info(f"   Complexity: {understanding.classification.complexity.value}")
            logger.info(f"   Sub-questions: {len(understanding.decomposition.sub_questions)}")
            logger.info(f"   Strategy: {understanding.generation_strategy}\n")

            # Stage 2: Retrieval
            logger.info("🔍 STAGE 2: MULTI-STEP RETRIEVAL")
            logger.info("-" * 70)
            retrieval_result = self.retriever.retrieve(
                query=query,
                query_understanding=understanding,
                max_chunks=max_chunks
            )
            logger.info(f"✅ Retrieval complete")
            logger.info(f"   Retrieved: {retrieval_result.total_chunks_retrieved} chunks")
            logger.info(f"   Final context: {retrieval_result.final_chunks} chunks")
            logger.info(f"   Sections: {retrieval_result.organized_context.unique_sections}")
            logger.info(f"   Time: {retrieval_result.retrieval_time:.2f}s\n")

            # Stage 3: Generation
            logger.info("✨ STAGE 3: ANSWER GENERATION")
            logger.info("-" * 70)
            answer = self.generator.generate(
                query=query,
                context=retrieval_result.organized_context,
                query_understanding=understanding
            )
            logger.info(f"✅ Answer generated")
            logger.info(f"   Length: {len(answer.answer.split())} words")
            logger.info(f"   Tokens: {answer.tokens_used}")
            logger.info(f"   Citations: {len(answer.citations)}")
            logger.info(f"   Images: {len(answer.images_used)}")
            logger.info(f"   Confidence: {answer.confidence:.2f}")
            logger.info(f"   Time: {answer.generation_time:.2f}s\n")

            # Stage 4: Validation
            if self.validator:
                logger.info("✓ STAGE 4: VALIDATION")
                logger.info("-" * 70)
                validation = self.validator.validate(
                    answer=answer,
                    query_understanding=understanding,
                    context=retrieval_result.organized_context
                )
                logger.info(f"{'✅' if validation.is_valid else '⚠️'} Validation {'passed' if validation.is_valid else 'has issues'}")
                logger.info(f"   Overall score: {validation.overall_score:.2f}")
                logger.info(f"   Completeness: {validation.completeness_score:.2f}")
                logger.info(f"   Formatting: {validation.formatting_score:.2f}")
                logger.info(f"   Issues: {len(validation.issues)}")
                logger.info(f"   Warnings: {len(validation.warnings)}\n")

                if validation.issues:
                    for issue in validation.issues:
                        logger.warning(f"   - {issue}")
            else:
                validation = None

            # Build result
            elapsed = time.time() - start_time

            result = PipelineResult(
                query=query,
                understanding=understanding,
                retrieval=retrieval_result,
                answer=answer,
                validation=validation,
                total_time=elapsed,
                success=success
            )

            logger.info("="*70)
            logger.info("✅ PIPELINE COMPLETE")
            logger.info(f"   Total time: {elapsed:.2f}s")
            logger.info("="*70 + "\n")

            return result

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            success = False
            raise


if __name__ == "__main__":
    """Test end-to-end pipeline."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    print("\n" + "="*70)
    print("🚀 END-TO-END RAG PIPELINE TEST")
    print("="*70 + "\n")

    # Initialize pipeline
    pipeline = EndToEndPipeline(
        use_reranking=True,
        enable_context_chaining=True,
        validate_responses=True
    )

    # Test query
    test_query = "How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"

    print(f"\n📝 Test Query:")
    print(f"   {test_query}\n")

    print("Processing...\n")

    # Process
    try:
        result = pipeline.process_query(test_query)

        # Display answer
        print("\n" + "="*70)
        print("📄 GENERATED ANSWER")
        print("="*70 + "\n")
        print(result.answer.answer)
        print("\n" + "="*70)

        # Display summary
        print("\n📊 PIPELINE SUMMARY")
        print("="*70)
        print(f"Query Type: {result.understanding.classification.query_type.value}")
        print(f"Complexity: {result.understanding.classification.complexity.value}")
        print(f"Sub-questions: {len(result.understanding.decomposition.sub_questions)}")
        print(f"\nRetrieval:")
        print(f"  - Total retrieved: {result.retrieval.total_chunks_retrieved} chunks")
        print(f"  - Final context: {result.retrieval.final_chunks} chunks")
        print(f"  - Unique sections: {result.retrieval.organized_context.unique_sections}")
        print(f"  - Time: {result.retrieval.retrieval_time:.2f}s")
        print(f"\nGeneration:")
        print(f"  - Length: {len(result.answer.answer.split())} words")
        print(f"  - Tokens: {result.answer.tokens_used}")
        print(f"  - Citations: {len(result.answer.citations)}")
        print(f"  - Confidence: {result.answer.confidence:.2f}")
        print(f"  - Time: {result.answer.generation_time:.2f}s")

        if result.validation:
            print(f"\nValidation:")
            print(f"  - Valid: {'✅ Yes' if result.validation.is_valid else '❌ No'}")
            print(f"  - Overall score: {result.validation.overall_score:.2f}")
            print(f"  - Completeness: {result.validation.completeness_score:.2f}")
            print(f"  - Formatting: {result.validation.formatting_score:.2f}")
            print(f"  - Citation: {result.validation.citation_score:.2f}")

            if result.validation.issues:
                print(f"\n  Issues:")
                for issue in result.validation.issues:
                    print(f"    - {issue}")

            if result.validation.warnings:
                print(f"\n  Warnings:")
                for warning in result.validation.warnings:
                    print(f"    - {warning}")

        print(f"\n⏱️  Total Time: {result.total_time:.2f}s")
        print("="*70)

        print("\n✅ Pipeline test complete!\n")

    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}\n")
        sys.exit(1)
