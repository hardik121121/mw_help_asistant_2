"""
Comprehensive System Analysis and Upgrade Recommendations.
Analyzes evaluation results to identify improvements needed.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List


def load_evaluation_results():
    """Load comprehensive evaluation results."""
    results_file = Path("tests/results/comprehensive_evaluation.json")
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_retrieval_performance(results: List[Dict]) -> Dict:
    """Analyze retrieval metrics across all queries."""

    precision_scores = [r['retrieval_metrics']['precision_at_10'] for r in results]
    recall_scores = [r['retrieval_metrics']['recall_at_10'] for r in results]
    mrr_scores = [r['retrieval_metrics']['mrr'] for r in results]
    coverage_scores = [r['retrieval_metrics']['coverage'] for r in results]

    # Identify low performers
    low_precision = [i+1 for i, p in enumerate(precision_scores) if p < 0.50]
    low_recall = [i+1 for i, r in enumerate(recall_scores) if r < 0.50]
    low_mrr = [i+1 for i, m in enumerate(mrr_scores) if m < 0.50]

    return {
        'precision': {
            'mean': statistics.mean(precision_scores),
            'median': statistics.median(precision_scores),
            'stdev': statistics.stdev(precision_scores),
            'min': min(precision_scores),
            'max': max(precision_scores),
            'low_performers': low_precision
        },
        'recall': {
            'mean': statistics.mean(recall_scores),
            'median': statistics.median(recall_scores),
            'stdev': statistics.stdev(recall_scores),
            'min': min(recall_scores),
            'max': max(recall_scores),
            'low_performers': low_recall
        },
        'mrr': {
            'mean': statistics.mean(mrr_scores),
            'median': statistics.median(mrr_scores),
            'stdev': statistics.stdev(mrr_scores),
            'min': min(mrr_scores),
            'max': max(mrr_scores),
            'low_performers': low_mrr
        },
        'coverage': {
            'mean': statistics.mean(coverage_scores),
            'median': statistics.median(coverage_scores),
            'stdev': statistics.stdev(coverage_scores),
            'min': min(coverage_scores),
            'max': max(coverage_scores)
        }
    }


def analyze_generation_performance(results: List[Dict]) -> Dict:
    """Analyze generation metrics across all queries."""

    overall_scores = [r['generation_metrics']['overall_score'] for r in results]
    completeness_scores = [r['generation_metrics']['completeness'] for r in results]
    word_counts = [r['generation_metrics']['word_count'] for r in results]

    # Identify issues
    low_quality = [i+1 for i, s in enumerate(overall_scores) if s < 0.85]
    too_short = [i+1 for i, w in enumerate(word_counts) if w < 300]
    too_long = [i+1 for i, w in enumerate(word_counts) if w > 700]

    return {
        'overall_score': {
            'mean': statistics.mean(overall_scores),
            'median': statistics.median(overall_scores),
            'stdev': statistics.stdev(overall_scores),
            'min': min(overall_scores),
            'max': max(overall_scores),
            'low_quality': low_quality
        },
        'completeness': {
            'mean': statistics.mean(completeness_scores),
            'median': statistics.median(completeness_scores),
            'perfect_count': sum(1 for c in completeness_scores if c == 1.0)
        },
        'word_count': {
            'mean': statistics.mean(word_counts),
            'median': statistics.median(word_counts),
            'stdev': statistics.stdev(word_counts),
            'min': min(word_counts),
            'max': max(word_counts),
            'too_short': too_short,
            'too_long': too_long
        }
    }


def analyze_performance_time(results: List[Dict]) -> Dict:
    """Analyze query processing time."""

    total_times = [r['total_time'] for r in results]
    slow_queries = [i+1 for i, t in enumerate(total_times) if t > 35]

    return {
        'mean': statistics.mean(total_times),
        'median': statistics.median(total_times),
        'stdev': statistics.stdev(total_times),
        'min': min(total_times),
        'max': max(total_times),
        'slow_queries': slow_queries
    }


def analyze_by_query_type(results: List[Dict]) -> Dict:
    """Analyze performance by query type."""

    by_type = {}
    for r in results:
        qtype = r['query_type']
        if qtype not in by_type:
            by_type[qtype] = []
        by_type[qtype].append(r)

    type_stats = {}
    for qtype, queries in by_type.items():
        type_stats[qtype] = {
            'count': len(queries),
            'avg_precision': statistics.mean([q['retrieval_metrics']['precision_at_10'] for q in queries]),
            'avg_generation_score': statistics.mean([q['generation_metrics']['overall_score'] for q in queries]),
            'avg_time': statistics.mean([q['total_time'] for q in queries])
        }

    return type_stats


def generate_recommendations(data: Dict) -> List[Dict]:
    """Generate actionable upgrade recommendations."""

    recommendations = []
    stats = data['statistics']

    # Retrieval recommendations
    if stats['retrieval']['avg_precision_at_10'] < 0.70:
        recommendations.append({
            'category': 'Retrieval - Precision',
            'priority': 'HIGH',
            'current': f"{stats['retrieval']['avg_precision_at_10']:.3f}",
            'target': '0.70',
            'issues': [
                'Precision@10 below target',
                'Too many irrelevant chunks in top results'
            ],
            'recommendations': [
                'Fine-tune embedding model on domain-specific data',
                'Implement query expansion with domain terms',
                'Adjust reranking weights (increase Cohere influence)',
                'Add metadata filtering for query types',
                'Consider using cross-encoder reranking'
            ]
        })

    if stats['retrieval']['avg_recall_at_10'] < 0.60:
        recommendations.append({
            'category': 'Retrieval - Recall',
            'priority': 'HIGH',
            'current': f"{stats['retrieval']['avg_recall_at_10']:.3f}",
            'target': '0.60',
            'issues': [
                'Recall@10 below target',
                'Missing relevant chunks in retrievals'
            ],
            'recommendations': [
                'Increase top_k in hybrid search (30 → 50)',
                'Improve query decomposition to capture all aspects',
                'Add query reformulation/paraphrasing',
                'Consider dense-sparse hybrid with ColBERT',
                'Implement HyDE (Hypothetical Document Embeddings)'
            ]
        })

    if stats['retrieval']['avg_mrr'] < 0.70:
        recommendations.append({
            'category': 'Retrieval - Ranking',
            'priority': 'MEDIUM',
            'current': f"{stats['retrieval']['avg_mrr']:.3f}",
            'target': '0.70',
            'issues': [
                'MRR below target',
                'Relevant chunks not ranked high enough'
            ],
            'recommendations': [
                'Train a custom reranking model on your data',
                'Adjust RRF fusion weights (test different ratios)',
                'Add LLM-based reranking as final step',
                'Implement learning-to-rank with user feedback'
            ]
        })

    if stats['retrieval']['avg_coverage'] < 0.80:
        recommendations.append({
            'category': 'Retrieval - Coverage',
            'priority': 'MEDIUM',
            'current': f"{stats['retrieval']['avg_coverage']:.3f}",
            'target': '0.80',
            'issues': [
                'Not all query topics being covered',
                'Missing topic identification in retrieval'
            ],
            'recommendations': [
                'Improve sub-question generation',
                'Add topic modeling/clustering',
                'Increase context window (20 → 30 chunks)',
                'Implement graph-based retrieval for related concepts'
            ]
        })

    # Generation recommendations
    if stats['generation']['avg_overall_score'] < 0.95:
        recommendations.append({
            'category': 'Generation - Quality',
            'priority': 'LOW',
            'current': f"{stats['generation']['avg_overall_score']:.3f}",
            'target': '0.95',
            'issues': [
                'Some answers below excellence threshold',
                'Formatting or citation issues'
            ],
            'recommendations': [
                'Fine-tune prompt templates',
                'Add few-shot examples for each query type',
                'Implement iterative refinement',
                'Use GPT-4 for complex queries',
                'Add structured output validation'
            ]
        })

    # Performance recommendations
    if stats['avg_query_time'] > 20:
        recommendations.append({
            'category': 'Performance - Speed',
            'priority': 'MEDIUM',
            'current': f"{stats['avg_query_time']:.2f}s",
            'target': '<15s',
            'issues': [
                'Average query time above target',
                'User experience impact'
            ],
            'recommendations': [
                'Implement query result caching',
                'Parallelize sub-question retrieval',
                'Use faster embedding model (reduce dimensions)',
                'Optimize Pinecone queries (use namespaces)',
                'Add async processing for non-critical paths',
                'Consider using Groq API Pro tier for faster inference'
            ]
        })

    # Cost optimization
    recommendations.append({
        'category': 'Cost Optimization',
        'priority': 'LOW',
        'current': '~$0.002/query',
        'target': '<$0.001/query',
        'issues': [
            'Cohere reranking adds cost per query',
            'Multiple embedding calls per query'
        ],
        'recommendations': [
            'Cache embeddings for common query patterns',
            'Use smaller embedding model for initial retrieval',
            'Batch reranking calls when possible',
            'Implement smart caching (Redis)',
            'Consider self-hosted reranking model'
        ]
    })

    # Infrastructure recommendations
    recommendations.append({
        'category': 'Infrastructure',
        'priority': 'MEDIUM',
        'current': 'Development setup',
        'target': 'Production-ready',
        'issues': [
            'No production deployment',
            'Missing monitoring and logging'
        ],
        'recommendations': [
            'Deploy with Docker containers',
            'Add Prometheus/Grafana for monitoring',
            'Implement structured logging (ELK stack)',
            'Set up CI/CD pipeline',
            'Add health checks and alerting',
            'Implement rate limiting and quotas',
            'Add A/B testing framework'
        ]
    })

    return recommendations


def print_analysis(data: Dict, retrieval_analysis: Dict, generation_analysis: Dict,
                  time_analysis: Dict, type_analysis: Dict, recommendations: List[Dict]):
    """Print comprehensive analysis report."""

    print('\n' + '='*80)
    print('📊 COMPREHENSIVE SYSTEM ANALYSIS & RECOMMENDATIONS')
    print('='*80)

    # Overall Summary
    print('\n' + '─'*80)
    print('1. OVERALL PERFORMANCE SUMMARY')
    print('─'*80)
    stats = data['statistics']
    print(f'  Total Queries Evaluated: {stats["num_total"]}')
    print(f'  Success Rate: {stats["success_rate"]*100:.1f}% ✅')
    print(f'  Average Processing Time: {stats["avg_query_time"]:.2f}s')
    print(f'  Total Processing Time: {stats["total_time"]:.1f}s ({stats["total_time"]/60:.1f} minutes)')

    # Retrieval Analysis
    print('\n' + '─'*80)
    print('2. RETRIEVAL PERFORMANCE ANALYSIS')
    print('─'*80)
    print(f'\n  📍 Precision@10:')
    print(f'     Mean: {retrieval_analysis["precision"]["mean"]:.3f} (Target: >0.70)')
    print(f'     Range: {retrieval_analysis["precision"]["min"]:.3f} - {retrieval_analysis["precision"]["max"]:.3f}')
    print(f'     Std Dev: {retrieval_analysis["precision"]["stdev"]:.3f}')
    print(f'     Low performers ({len(retrieval_analysis["precision"]["low_performers"])}): {retrieval_analysis["precision"]["low_performers"][:5]}...' if len(retrieval_analysis["precision"]["low_performers"]) > 5 else f'     Low performers: {retrieval_analysis["precision"]["low_performers"]}')

    print(f'\n  📍 Recall@10:')
    print(f'     Mean: {retrieval_analysis["recall"]["mean"]:.3f} (Target: >0.60)')
    print(f'     Range: {retrieval_analysis["recall"]["min"]:.3f} - {retrieval_analysis["recall"]["max"]:.3f}')
    print(f'     Std Dev: {retrieval_analysis["recall"]["stdev"]:.3f}')
    print(f'     Low performers: {retrieval_analysis["recall"]["low_performers"][:5]}...' if len(retrieval_analysis["recall"]["low_performers"]) > 5 else f'     Low performers: {retrieval_analysis["recall"]["low_performers"]}')

    print(f'\n  📍 MRR (Mean Reciprocal Rank):')
    print(f'     Mean: {retrieval_analysis["mrr"]["mean"]:.3f} (Target: >0.70)')
    print(f'     Range: {retrieval_analysis["mrr"]["min"]:.3f} - {retrieval_analysis["mrr"]["max"]:.3f}')
    print(f'     Std Dev: {retrieval_analysis["mrr"]["stdev"]:.3f}')

    print(f'\n  📍 Coverage:')
    print(f'     Mean: {retrieval_analysis["coverage"]["mean"]:.3f} (Target: >0.80)')
    print(f'     Range: {retrieval_analysis["coverage"]["min"]:.3f} - {retrieval_analysis["coverage"]["max"]:.3f}')

    # Generation Analysis
    print('\n' + '─'*80)
    print('3. GENERATION QUALITY ANALYSIS')
    print('─'*80)
    print(f'\n  ✨ Overall Quality Score:')
    print(f'     Mean: {generation_analysis["overall_score"]["mean"]:.3f} (Target: >0.95)')
    print(f'     Range: {generation_analysis["overall_score"]["min"]:.3f} - {generation_analysis["overall_score"]["max"]:.3f}')
    print(f'     Std Dev: {generation_analysis["overall_score"]["stdev"]:.3f}')
    print(f'     Below excellence (<0.85): {len(generation_analysis["overall_score"]["low_quality"])} queries')

    print(f'\n  ✨ Completeness:')
    print(f'     Mean: {generation_analysis["completeness"]["mean"]:.3f}')
    print(f'     Perfect (1.0): {generation_analysis["completeness"]["perfect_count"]}/{stats["num_total"]} queries')

    print(f'\n  ✨ Word Count:')
    print(f'     Mean: {generation_analysis["word_count"]["mean"]:.0f} words')
    print(f'     Range: {generation_analysis["word_count"]["min"]}-{generation_analysis["word_count"]["max"]} words')
    print(f'     Too short (<300): {len(generation_analysis["word_count"]["too_short"])} queries')
    print(f'     Too long (>700): {len(generation_analysis["word_count"]["too_long"])} queries')

    # Performance Analysis
    print('\n' + '─'*80)
    print('4. PERFORMANCE ANALYSIS')
    print('─'*80)
    print(f'  ⚡ Query Processing Time:')
    print(f'     Mean: {time_analysis["mean"]:.2f}s (Target: <15s)')
    print(f'     Median: {time_analysis["median"]:.2f}s')
    print(f'     Range: {time_analysis["min"]:.2f}s - {time_analysis["max"]:.2f}s')
    print(f'     Std Dev: {time_analysis["stdev"]:.2f}s')
    print(f'     Slow queries (>35s): {len(time_analysis["slow_queries"])} queries')

    # Query Type Analysis
    print('\n' + '─'*80)
    print('5. PERFORMANCE BY QUERY TYPE')
    print('─'*80)
    for qtype, qstats in sorted(type_analysis.items(), key=lambda x: x[1]['avg_precision'], reverse=True):
        print(f'\n  📋 {qtype}:')
        print(f'     Count: {qstats["count"]} queries')
        print(f'     Avg Precision: {qstats["avg_precision"]:.3f}')
        print(f'     Avg Generation Score: {qstats["avg_generation_score"]:.3f}')
        print(f'     Avg Time: {qstats["avg_time"]:.2f}s')

    # Recommendations
    print('\n' + '='*80)
    print('6. UPGRADE RECOMMENDATIONS')
    print('='*80)

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    sorted_recs = sorted(recommendations, key=lambda x: priority_order[x['priority']])

    for i, rec in enumerate(sorted_recs, 1):
        print(f'\n{"-"*80}')
        print(f'{i}. {rec["category"]} [Priority: {rec["priority"]}]')
        print(f'{"-"*80}')
        print(f'  Current: {rec["current"]}')
        print(f'  Target: {rec["target"]}')

        if rec.get('issues'):
            print(f'\n  ⚠️  Issues:')
            for issue in rec['issues']:
                print(f'     • {issue}')

        print(f'\n  💡 Recommendations:')
        for r in rec['recommendations']:
            print(f'     • {r}')

    print('\n' + '='*80)
    print('7. IMMEDIATE ACTION ITEMS')
    print('='*80)

    high_priority = [r for r in sorted_recs if r['priority'] == 'HIGH']
    if high_priority:
        print('\n  🔴 High Priority (Address First):')
        for rec in high_priority:
            print(f'     1. {rec["category"]}: {rec["recommendations"][0]}')

    medium_priority = [r for r in sorted_recs if r['priority'] == 'MEDIUM']
    if medium_priority:
        print('\n  🟡 Medium Priority (Plan for Q1):')
        for rec in medium_priority:
            print(f'     • {rec["category"]}: {rec["recommendations"][0]}')

    print('\n' + '='*80)
    print('📁 Full results available in: tests/results/comprehensive_evaluation.json')
    print('='*80 + '\n')


def main():
    """Run comprehensive analysis."""
    print('\n🚀 Running Comprehensive System Analysis...\n')

    # Load data
    data = load_evaluation_results()
    results = [r for r in data['results'] if r['success']]

    # Perform analyses
    retrieval_analysis = analyze_retrieval_performance(results)
    generation_analysis = analyze_generation_performance(results)
    time_analysis = analyze_performance_time(results)
    type_analysis = analyze_by_query_type(results)
    recommendations = generate_recommendations(data)

    # Print report
    print_analysis(data, retrieval_analysis, generation_analysis,
                   time_analysis, type_analysis, recommendations)

    # Save recommendations to file
    output_file = Path("tests/results/system_recommendations.json")
    with open(output_file, 'w') as f:
        json.dump({
            'retrieval_analysis': retrieval_analysis,
            'generation_analysis': generation_analysis,
            'time_analysis': time_analysis,
            'type_analysis': type_analysis,
            'recommendations': recommendations
        }, f, indent=2)

    print(f'✅ Detailed recommendations saved to: {output_file}\n')


if __name__ == "__main__":
    main()
