"""
Continue evaluation from query 7 onwards.
This script resumes evaluation after API rate limits.
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


def load_existing_results():
    """Load existing evaluation results."""
    settings = get_settings()
    results_file = settings.evaluation_output_dir / "comprehensive_evaluation.json"

    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def continue_evaluation(start_query_id=7):
    """
    Continue evaluation from a specific query ID.

    Args:
        start_query_id: Query ID to start from (1-indexed)
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"RESUMING EVALUATION FROM QUERY {start_query_id}")
    logger.info(f"{'='*70}\n")

    # Load existing results
    existing_data = load_existing_results()
    if existing_data:
        logger.info(f"✅ Loaded existing results: {len(existing_data['results'])} queries")
        existing_results = existing_data['results']
    else:
        logger.info("⚠️  No existing results found, starting fresh")
        existing_results = []

    # Load all test queries
    settings = get_settings()
    with open(settings.test_queries_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_queries = data.get('queries', [])

    # Filter to queries starting from start_query_id
    queries_to_evaluate = [q for q in all_queries if q['id'] >= start_query_id]

    logger.info(f"📋 Total queries: {len(all_queries)}")
    logger.info(f"📋 Queries to evaluate: {len(queries_to_evaluate)} (IDs {start_query_id}-{all_queries[-1]['id']})")
    logger.info(f"📋 Already completed: {len(existing_results)}\n")

    # Create evaluator
    evaluator = ComprehensiveEvaluator(test_limit=None)
    evaluator.test_queries = queries_to_evaluate

    logger.info(f"Starting evaluation of remaining queries...\n")

    # Run evaluation on remaining queries
    new_results_data = evaluator.evaluate_all()

    # Merge results
    all_results = existing_results + new_results_data['results']

    # Recalculate overall statistics
    successful = [r for r in all_results if r['success']]
    num_successful = len(successful)
    num_total = len(all_results)

    if num_successful > 0:
        # Recalculate all statistics for combined results
        stats = evaluator._calculate_statistics(
            [type('obj', (object,), r) for r in all_results],
            sum(r['total_time'] for r in all_results)
        )
    else:
        stats = {
            'num_total': num_total,
            'num_successful': 0,
            'num_failed': num_total,
            'success_rate': 0.0,
            'total_time': sum(r['total_time'] for r in all_results),
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

    # Save combined results
    output_file = settings.evaluation_output_dir / "comprehensive_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*70}")
    logger.info("✅ COMBINED EVALUATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total Queries: {stats['num_total']}")
    logger.info(f"  Successful: {stats['num_successful']}")
    logger.info(f"  Failed: {stats['num_failed']}")
    logger.info(f"  Success Rate: {stats['success_rate']*100:.1f}%")
    logger.info(f"\n✅ Results saved to: {output_file}")
    logger.info(f"{'='*70}\n")

    return final_output


if __name__ == "__main__":
    try:
        results = continue_evaluation(start_query_id=7)
        print("\n✅ Evaluation continuation complete!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
