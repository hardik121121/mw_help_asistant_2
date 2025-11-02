"""
Phase 4 Test Script: Multi-Step Retrieval System
Tests all retrieval components with complex queries.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict

from src.retrieval.multi_step_retriever import MultiStepRetriever, RetrievalResult
from src.query.query_understanding import QueryUnderstandingEngine
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase4Tester:
    """Test suite for Phase 4 retrieval components."""

    def __init__(self):
        """Initialize tester."""
        self.settings = get_settings()
        self.query_system = QueryUnderstandingEngine()
        self.retriever = MultiStepRetriever(
            use_reranking=True,
            enable_context_chaining=True
        )

        # Load test queries
        test_queries_path = self.settings.test_queries_path
        with open(test_queries_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict formats
            if isinstance(data, dict):
                self.test_queries = data.get('queries', list(data.values()))
            else:
                self.test_queries = data

        logger.info(f"Loaded {len(self.test_queries)} test queries")

    def test_single_query(self, query_data: Dict) -> Dict:
        """
        Test retrieval for a single query.

        Args:
            query_data: Query dict from test_queries.json

        Returns:
            Test result dict
        """
        query = query_data['query']
        query_id = query_data['id']

        logger.info(f"\n{'='*70}")
        logger.info(f"TESTING QUERY {query_id}")
        logger.info(f"{'='*70}")
        logger.info(f"Query: {query}")
        logger.info(f"Type: {query_data['type']}")
        logger.info(f"Complexity: {query_data['complexity']}")
        logger.info(f"Topics: {', '.join(query_data['topics'])}\n")

        start_time = time.time()

        try:
            # Step 1: Understand query
            logger.info("Step 1: Understanding query...")
            understanding = self.query_system.understand(query)

            logger.info(f"  Classified as: {understanding.classification.query_type}")
            logger.info(f"  Complexity: {understanding.classification.complexity}")
            logger.info(f"  Sub-questions: {len(understanding.decomposition.sub_questions)}")

            # Step 2: Retrieve
            logger.info("\nStep 2: Multi-step retrieval...")
            retrieval_result = self.retriever.retrieve(
                query=query,
                query_understanding=understanding,
                max_chunks=20
            )

            # Step 3: Evaluate
            logger.info("\nStep 3: Evaluating results...")
            evaluation = self._evaluate_retrieval(
                query_data,
                retrieval_result
            )

            elapsed = time.time() - start_time

            # Build result
            result = {
                'query_id': query_id,
                'query': query,
                'success': True,
                'num_sub_questions': len(understanding.decomposition.sub_questions),
                'total_chunks_retrieved': retrieval_result.total_chunks_retrieved,
                'final_chunks': retrieval_result.final_chunks,
                'unique_sections': retrieval_result.organized_context.unique_sections,
                'page_range': retrieval_result.organized_context.page_range,
                'has_images': retrieval_result.organized_context.has_images,
                'has_tables': retrieval_result.organized_context.has_tables,
                'retrieval_time': retrieval_result.retrieval_time,
                'total_time': elapsed,
                'evaluation': evaluation,
                'topic_coverage': list(retrieval_result.organized_context.topic_groups.keys())
            }

            logger.info(f"\n✅ Query {query_id} tested successfully")
            logger.info(f"   Retrieved: {result['final_chunks']} chunks")
            logger.info(f"   Sections: {result['unique_sections']}")
            logger.info(f"   Time: {result['total_time']:.2f}s")

            return result

        except Exception as e:
            logger.error(f"❌ Query {query_id} failed: {e}", exc_info=True)
            return {
                'query_id': query_id,
                'query': query,
                'success': False,
                'error': str(e)
            }

    def _evaluate_retrieval(self,
                           query_data: Dict,
                           retrieval_result: RetrievalResult) -> Dict:
        """
        Evaluate retrieval quality.

        Args:
            query_data: Original query data
            retrieval_result: Retrieval results

        Returns:
            Evaluation dict
        """
        evaluation = {
            'topic_coverage': 0.0,
            'section_diversity': 0.0,
            'has_all_components': False,
            'estimated_quality': 'unknown'
        }

        # Check topic coverage
        expected_topics = query_data.get('topics', [])
        retrieved_topics = list(retrieval_result.organized_context.topic_groups.keys())

        if expected_topics:
            # Simple keyword matching
            covered = 0
            for topic in expected_topics:
                topic_keywords = topic.lower().split()
                for retrieved_topic in retrieved_topics:
                    if any(kw in retrieved_topic.lower() for kw in topic_keywords):
                        covered += 1
                        break

            evaluation['topic_coverage'] = covered / len(expected_topics)

        # Section diversity (more unique sections = better)
        num_sections = retrieval_result.organized_context.unique_sections
        evaluation['section_diversity'] = min(num_sections / 10, 1.0)

        # Check for expected components
        expected_components = query_data.get('expected_components', [])
        evaluation['has_all_components'] = len(expected_components) > 0

        # Estimate overall quality
        if evaluation['topic_coverage'] > 0.8 and num_sections >= 3:
            evaluation['estimated_quality'] = 'excellent'
        elif evaluation['topic_coverage'] > 0.6 and num_sections >= 2:
            evaluation['estimated_quality'] = 'good'
        elif evaluation['topic_coverage'] > 0.4:
            evaluation['estimated_quality'] = 'fair'
        else:
            evaluation['estimated_quality'] = 'needs_improvement'

        return evaluation

    def test_all_queries(self, limit: int = 5) -> Dict:
        """
        Test retrieval on multiple queries.

        Args:
            limit: Maximum number of queries to test

        Returns:
            Aggregated results
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 4 COMPREHENSIVE TEST")
        logger.info("="*70 + "\n")

        results = []
        test_queries = self.test_queries[:limit]

        for i, query_data in enumerate(test_queries, 1):
            logger.info(f"\n### Testing query {i}/{len(test_queries)} ###\n")
            result = self.test_single_query(query_data)
            results.append(result)

            # Pause between queries
            if i < len(test_queries):
                time.sleep(2)

        # Aggregate statistics
        stats = self._compute_statistics(results)

        # Save results
        self._save_results(results, stats)

        # Print summary
        self._print_summary(stats)

        return {
            'results': results,
            'statistics': stats
        }

    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute aggregate statistics."""
        successful = [r for r in results if r.get('success', False)]
        num_total = len(results)
        num_successful = len(successful)

        if not successful:
            return {'success_rate': 0.0, 'num_tested': num_total}

        stats = {
            'num_tested': num_total,
            'num_successful': num_successful,
            'success_rate': num_successful / num_total,
            'avg_retrieval_time': sum(r['retrieval_time'] for r in successful) / num_successful,
            'avg_total_time': sum(r['total_time'] for r in successful) / num_successful,
            'avg_final_chunks': sum(r['final_chunks'] for r in successful) / num_successful,
            'avg_sections': sum(r['unique_sections'] for r in successful) / num_successful,
            'avg_topic_coverage': sum(r['evaluation']['topic_coverage'] for r in successful) / num_successful,
            'quality_distribution': {
                'excellent': sum(1 for r in successful if r['evaluation']['estimated_quality'] == 'excellent'),
                'good': sum(1 for r in successful if r['evaluation']['estimated_quality'] == 'good'),
                'fair': sum(1 for r in successful if r['evaluation']['estimated_quality'] == 'fair'),
                'needs_improvement': sum(1 for r in successful if r['evaluation']['estimated_quality'] == 'needs_improvement')
            }
        }

        return stats

    def _save_results(self, results: List[Dict], stats: Dict):
        """Save test results to file."""
        output_dir = self.settings.evaluation_output_dir
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "phase4_test_results.json"

        output = {
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': stats,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✅ Results saved to: {output_file}")

    def _print_summary(self, stats: Dict):
        """Print test summary."""
        print("\n" + "="*70)
        print("PHASE 4 TEST SUMMARY")
        print("="*70)
        print(f"\n📊 Overall Statistics:")
        print(f"  Queries Tested: {stats['num_tested']}")
        print(f"  Successful: {stats['num_successful']}")
        print(f"  Success Rate: {stats['success_rate']*100:.1f}%")

        if stats['num_successful'] > 0:
            print(f"\n⚡ Performance:")
            print(f"  Avg Retrieval Time: {stats['avg_retrieval_time']:.2f}s")
            print(f"  Avg Total Time: {stats['avg_total_time']:.2f}s")

            print(f"\n📚 Retrieval Quality:")
            print(f"  Avg Final Chunks: {stats['avg_final_chunks']:.1f}")
            print(f"  Avg Unique Sections: {stats['avg_sections']:.1f}")
            print(f"  Avg Topic Coverage: {stats['avg_topic_coverage']*100:.1f}%")

            print(f"\n⭐ Quality Distribution:")
            for quality, count in stats['quality_distribution'].items():
                print(f"  {quality.capitalize()}: {count}")

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    """Run Phase 4 tests."""
    print("\n" + "="*70)
    print("🚀 PHASE 4: MULTI-STEP RETRIEVAL SYSTEM - TESTING")
    print("="*70 + "\n")

    # Initialize tester
    tester = Phase4Tester()

    # Test on 5 complex queries
    print("Testing retrieval on 5 complex queries...\n")
    results = tester.test_all_queries(limit=5)

    print("\n✅ Phase 4 testing complete!")
    print("\nResults saved to: tests/results/phase4_test_results.json")
    print("\n" + "="*70 + "\n")
