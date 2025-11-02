"""
Comprehensive System Evaluation.
Tests the complete RAG pipeline on all test queries.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict

from src.generation.end_to_end_pipeline import EndToEndPipeline, PipelineResult
from src.evaluation.retrieval_metrics import RetrievalMetricsCalculator, RetrievalMetrics
from src.evaluation.generation_metrics import GenerationMetricsCalculator, GenerationMetrics
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryEvaluationResult:
    """Evaluation result for a single query."""
    query_id: int
    query: str
    query_type: str
    complexity: str
    pipeline_result: Dict
    retrieval_metrics: Dict
    generation_metrics: Dict
    total_time: float
    success: bool
    error: str = ""


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation of the RAG system.

    Evaluates:
    - All 30 test queries
    - Retrieval quality metrics
    - Generation quality metrics
    - Performance metrics
    - Cost metrics
    """

    def __init__(self, test_limit: int = None):
        """
        Initialize evaluator.

        Args:
            test_limit: Limit number of queries to test (None = all)
        """
        self.settings = get_settings()
        self.test_limit = test_limit

        # Initialize pipeline
        logger.info("Initializing end-to-end pipeline...")
        self.pipeline = EndToEndPipeline(
            use_reranking=True,
            enable_context_chaining=True,
            validate_responses=True
        )

        # Initialize metrics calculators
        self.retrieval_metrics_calc = RetrievalMetricsCalculator()
        self.generation_metrics_calc = GenerationMetricsCalculator()

        # Load test queries
        test_queries_path = self.settings.test_queries_path
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.test_queries = data.get('queries', [])

        if self.test_limit:
            self.test_queries = self.test_queries[:self.test_limit]

        logger.info(f"Loaded {len(self.test_queries)} test queries")

    def evaluate_all(self) -> Dict:
        """
        Evaluate all test queries.

        Returns:
            Dict with results and statistics
        """
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE SYSTEM EVALUATION")
        logger.info("="*70)
        logger.info(f"Evaluating {len(self.test_queries)} queries\n")

        results = []
        start_time = time.time()

        for i, query_data in enumerate(self.test_queries, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"QUERY {i}/{len(self.test_queries)}")
            logger.info(f"{'='*70}")

            result = self.evaluate_single_query(query_data, i)
            results.append(result)

            # Brief pause between queries
            if i < len(self.test_queries):
                time.sleep(1)

        total_time = time.time() - start_time

        # Calculate aggregate statistics
        stats = self._calculate_statistics(results, total_time)

        # Save results
        output = {
            'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_queries': len(self.test_queries),
            'statistics': stats,
            'results': [asdict(r) for r in results]
        }

        self._save_results(output)

        # Print summary
        self._print_summary(stats)

        return output

    def evaluate_single_query(self,
                             query_data: Dict,
                             query_num: int) -> QueryEvaluationResult:
        """
        Evaluate a single query.

        Args:
            query_data: Query dict from test_queries.json
            query_num: Query number for logging

        Returns:
            QueryEvaluationResult
        """
        query = query_data['query']
        query_id = query_data['id']

        logger.info(f"Query: {query}")
        logger.info(f"Type: {query_data['type']}")
        logger.info(f"Complexity: {query_data['complexity']}\n")

        start_time = time.time()

        try:
            # Run pipeline
            pipeline_result = self.pipeline.process_query(query)

            # Calculate retrieval metrics
            logger.info("\nCalculating retrieval metrics...")
            retrieval_metrics = self.retrieval_metrics_calc.calculate_metrics(
                retrieved_chunks=pipeline_result.retrieval.organized_context.chunks,
                expected_topics=query_data.get('topics', []),
                query_data=query_data
            )

            # Calculate generation metrics
            logger.info("\nCalculating generation metrics...")
            sub_questions = [
                sq.question
                for sq in pipeline_result.understanding.decomposition.sub_questions
            ]

            generation_metrics = self.generation_metrics_calc.calculate_metrics(
                answer_text=pipeline_result.answer.answer,
                expected_topics=query_data.get('topics', []),
                sub_questions=sub_questions,
                has_citations=len(pipeline_result.answer.citations) > 0
            )

            elapsed = time.time() - start_time

            result = QueryEvaluationResult(
                query_id=query_id,
                query=query,
                query_type=query_data['type'],
                complexity=query_data['complexity'],
                pipeline_result=pipeline_result.to_dict(),
                retrieval_metrics={
                    'precision_at_10': retrieval_metrics.precision_at_k.get(10, 0),
                    'recall_at_10': retrieval_metrics.recall_at_k.get(10, 0),
                    'mrr': retrieval_metrics.mrr,
                    'map': retrieval_metrics.map_score,
                    'ndcg_at_10': retrieval_metrics.ndcg_at_k.get(10, 0),
                    'coverage': retrieval_metrics.coverage,
                    'diversity': retrieval_metrics.diversity,
                    'total_retrieved': retrieval_metrics.total_retrieved,
                    'relevant_retrieved': retrieval_metrics.relevant_retrieved
                },
                generation_metrics={
                    'overall_score': generation_metrics.overall_score,
                    'completeness': generation_metrics.completeness_score,
                    'coherence': generation_metrics.coherence_score,
                    'formatting': generation_metrics.formatting_score,
                    'citation': generation_metrics.citation_score,
                    'length': generation_metrics.length_score,
                    'keyword_coverage': generation_metrics.keyword_coverage,
                    'word_count': generation_metrics.word_count,
                    'has_headings': generation_metrics.has_headings,
                    'has_lists': generation_metrics.has_lists
                },
                total_time=elapsed,
                success=True
            )

            logger.info(f"\n✅ Query {query_num} evaluation complete")
            logger.info(f"   Total time: {elapsed:.2f}s")
            logger.info(f"   Retrieval P@10: {retrieval_metrics.precision_at_k.get(10, 0):.3f}")
            logger.info(f"   Generation score: {generation_metrics.overall_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"❌ Query {query_num} failed: {e}", exc_info=True)

            elapsed = time.time() - start_time

            return QueryEvaluationResult(
                query_id=query_id,
                query=query,
                query_type=query_data['type'],
                complexity=query_data['complexity'],
                pipeline_result={},
                retrieval_metrics={},
                generation_metrics={},
                total_time=elapsed,
                success=False,
                error=str(e)
            )

    def _calculate_statistics(self,
                             results: List[QueryEvaluationResult],
                             total_time: float) -> Dict:
        """Calculate aggregate statistics."""
        successful = [r for r in results if r.success]
        num_successful = len(successful)
        num_total = len(results)

        if num_successful == 0:
            return {
                'success_rate': 0.0,
                'num_total': num_total,
                'num_successful': 0,
                'num_failed': num_total,
                'total_time': total_time,
                'avg_query_time': 0,
                'retrieval': {},
                'generation': {}
            }

        # Retrieval stats
        avg_precision_10 = sum(
            r.retrieval_metrics.get('precision_at_10', 0)
            for r in successful
        ) / num_successful

        avg_recall_10 = sum(
            r.retrieval_metrics.get('recall_at_10', 0)
            for r in successful
        ) / num_successful

        avg_mrr = sum(
            r.retrieval_metrics.get('mrr', 0)
            for r in successful
        ) / num_successful

        avg_coverage = sum(
            r.retrieval_metrics.get('coverage', 0)
            for r in successful
        ) / num_successful

        avg_diversity = sum(
            r.retrieval_metrics.get('diversity', 0)
            for r in successful
        ) / num_successful

        # Generation stats
        avg_generation_score = sum(
            r.generation_metrics.get('overall_score', 0)
            for r in successful
        ) / num_successful

        avg_completeness = sum(
            r.generation_metrics.get('completeness', 0)
            for r in successful
        ) / num_successful

        avg_word_count = sum(
            r.generation_metrics.get('word_count', 0)
            for r in successful
        ) / num_successful

        # Performance stats
        avg_query_time = sum(r.total_time for r in successful) / num_successful

        # Quality distribution
        excellent_count = sum(
            1 for r in successful
            if r.generation_metrics.get('overall_score', 0) >= 0.85
        )
        good_count = sum(
            1 for r in successful
            if 0.70 <= r.generation_metrics.get('overall_score', 0) < 0.85
        )
        fair_count = sum(
            1 for r in successful
            if 0.50 <= r.generation_metrics.get('overall_score', 0) < 0.70
        )
        poor_count = sum(
            1 for r in successful
            if r.generation_metrics.get('overall_score', 0) < 0.50
        )

        return {
            'num_total': num_total,
            'num_successful': num_successful,
            'num_failed': num_total - num_successful,
            'success_rate': num_successful / num_total,
            'total_time': total_time,
            'avg_query_time': avg_query_time,
            'retrieval': {
                'avg_precision_at_10': avg_precision_10,
                'avg_recall_at_10': avg_recall_10,
                'avg_mrr': avg_mrr,
                'avg_coverage': avg_coverage,
                'avg_diversity': avg_diversity
            },
            'generation': {
                'avg_overall_score': avg_generation_score,
                'avg_completeness': avg_completeness,
                'avg_word_count': avg_word_count,
                'quality_distribution': {
                    'excellent': excellent_count,
                    'good': good_count,
                    'fair': fair_count,
                    'poor': poor_count
                }
            }
        }

    def _save_results(self, output: Dict):
        """Save evaluation results to file."""
        output_dir = self.settings.evaluation_output_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        output_file = output_dir / "comprehensive_evaluation.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ Results saved to: {output_file}")

    def _print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("="*70)

        print(f"\n📊 Overall Statistics:")
        print(f"  Total Queries: {stats['num_total']}")
        print(f"  Successful: {stats['num_successful']}")
        print(f"  Failed: {stats['num_failed']}")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"  Total Time: {stats['total_time']:.1f}s")
        print(f"  Avg Time per Query: {stats['avg_query_time']:.2f}s")

        print(f"\n🔍 Retrieval Metrics:")
        print(f"  Precision@10: {stats['retrieval']['avg_precision_at_10']:.3f}")
        print(f"  Recall@10: {stats['retrieval']['avg_recall_at_10']:.3f}")
        print(f"  MRR: {stats['retrieval']['avg_mrr']:.3f}")
        print(f"  Coverage: {stats['retrieval']['avg_coverage']:.3f}")
        print(f"  Diversity: {stats['retrieval']['avg_diversity']:.3f}")

        print(f"\n✨ Generation Metrics:")
        print(f"  Overall Score: {stats['generation']['avg_overall_score']:.3f}")
        print(f"  Completeness: {stats['generation']['avg_completeness']:.3f}")
        print(f"  Avg Word Count: {stats['generation']['avg_word_count']:.0f}")

        print(f"\n⭐ Quality Distribution:")
        dist = stats['generation']['quality_distribution']
        print(f"  Excellent (≥0.85): {dist['excellent']}")
        print(f"  Good (0.70-0.85): {dist['good']}")
        print(f"  Fair (0.50-0.70): {dist['fair']}")
        print(f"  Poor (<0.50): {dist['poor']}")

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    """Run comprehensive evaluation."""
    import sys

    print("\n" + "="*70)
    print("🚀 COMPREHENSIVE SYSTEM EVALUATION")
    print("="*70)
    print("\nThis will evaluate the RAG pipeline on all test queries.")
    print("This may take 10-20 minutes depending on the number of queries.\n")

    # Ask for confirmation
    response = input("Run evaluation on how many queries? (default: 5, 'all' for all 30): ").strip()

    if response.lower() == 'all':
        test_limit = None
        print(f"\n✓ Running on ALL test queries...")
    elif response.isdigit():
        test_limit = int(response)
        print(f"\n✓ Running on first {test_limit} queries...")
    else:
        test_limit = 5
        print(f"\n✓ Running on first 5 queries (default)...")

    print("\nStarting evaluation...\n")

    try:
        evaluator = ComprehensiveEvaluator(test_limit=test_limit)
        results = evaluator.evaluate_all()

        print("\n✅ Comprehensive evaluation complete!")
        print(f"\nResults saved to: tests/results/comprehensive_evaluation.json")
        print("\n" + "="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
