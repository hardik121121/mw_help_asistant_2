"""
Finish evaluation for remaining failed queries.
"""

import json
import logging
import time
from pathlib import Path

from src.evaluation.comprehensive_evaluation import ComprehensiveEvaluator
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def finish_evaluation():
    """
    Complete evaluation for remaining failed queries.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"FINISHING EVALUATION - REMAINING QUERIES")
    logger.info(f"{'='*70}\n")

    # Load existing results
    settings = get_settings()
    results_file = settings.evaluation_output_dir / "comprehensive_evaluation.json"

    with open(results_file, 'r', encoding='utf-8') as f:
        existing_data = json.load(f)

    existing_results = {r['query_id']: r for r in existing_data['results']}
    failed_ids = [qid for qid, r in existing_results.items() if not r['success']]

    logger.info(f"✅ Loaded existing results: {len(existing_results)} queries")
    logger.info(f"❌ Failed queries to retry: {failed_ids}\n")

    # Load all test queries
    with open(settings.test_queries_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_queries = data.get('queries', [])

    # Filter to only failed queries
    queries_to_evaluate = [q for q in all_queries if q['id'] in failed_ids]

    logger.info(f"📋 Queries to evaluate: {len(queries_to_evaluate)}\n")

    # Create evaluator
    evaluator = ComprehensiveEvaluator(test_limit=None)
    evaluator.test_queries = queries_to_evaluate

    logger.info(f"Starting evaluation of remaining queries...\n")

    # Run evaluation
    new_results_data = evaluator.evaluate_all()

    # Merge results - replace failed queries with new results
    for new_result in new_results_data['results']:
        existing_results[new_result['query_id']] = new_result

    # Get all results sorted by ID
    all_results = sorted(existing_results.values(), key=lambda x: x['query_id'])

    # Recalculate statistics
    successful = [r for r in all_results if r['success']]
    num_successful = len(successful)
    num_total = len(all_results)
    total_time = sum(r['total_time'] for r in all_results)

    if num_successful > 0:
        # Retrieval stats
        avg_precision_10 = sum(r['retrieval_metrics']['precision_at_10'] for r in successful) / num_successful
        avg_recall_10 = sum(r['retrieval_metrics']['recall_at_10'] for r in successful) / num_successful
        avg_mrr = sum(r['retrieval_metrics']['mrr'] for r in successful) / num_successful
        avg_coverage = sum(r['retrieval_metrics']['coverage'] for r in successful) / num_successful
        avg_diversity = sum(r['retrieval_metrics']['diversity'] for r in successful) / num_successful

        # Generation stats
        avg_generation_score = sum(r['generation_metrics']['overall_score'] for r in successful) / num_successful
        avg_completeness = sum(r['generation_metrics']['completeness'] for r in successful) / num_successful
        avg_word_count = sum(r['generation_metrics']['word_count'] for r in successful) / num_successful
        avg_query_time = sum(r['total_time'] for r in successful) / num_successful

        # Quality distribution
        excellent_count = sum(1 for r in successful if r['generation_metrics']['overall_score'] >= 0.85)
        good_count = sum(1 for r in successful if 0.70 <= r['generation_metrics']['overall_score'] < 0.85)
        fair_count = sum(1 for r in successful if 0.50 <= r['generation_metrics']['overall_score'] < 0.70)
        poor_count = sum(1 for r in successful if r['generation_metrics']['overall_score'] < 0.50)

        stats = {
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
    else:
        stats = {
            'num_total': num_total,
            'num_successful': 0,
            'num_failed': num_total,
            'success_rate': 0.0,
            'total_time': total_time,
            'avg_query_time': 0,
            'retrieval': {},
            'generation': {}
        }

    # Create final output
    final_output = {
        'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_queries': num_total,
        'statistics': stats,
        'results': all_results
    }

    # Save final results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*70}")
    logger.info("✅ FINAL COMPLETE EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total Queries: {stats['num_total']}")
    logger.info(f"  Successful: {stats['num_successful']}")
    logger.info(f"  Failed: {stats['num_failed']}")
    logger.info(f"  Success Rate: {stats['success_rate']*100:.1f}%")
    logger.info(f"\n✅ Results saved to: {results_file}")
    logger.info(f"{'='*70}\n")

    return final_output


if __name__ == "__main__":
    try:
        results = finish_evaluation()
        print("\n✅ Evaluation completion successful!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
