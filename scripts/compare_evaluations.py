"""Compare two evaluation results."""
import json
import sys
from pathlib import Path


def compare_evaluations(baseline_path: str, new_path: str):
    """
    Compare two evaluation results and show improvements.

    Args:
        baseline_path: Path to baseline evaluation JSON
        new_path: Path to new evaluation JSON
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    print("\n" + "="*80)
    print(" 📊 EVALUATION COMPARISON")
    print("="*80)
    print(f"Baseline: {baseline_path}")
    print(f"New:      {new_path}")
    print("="*80 + "\n")

    # Compare overall statistics
    metrics = [
        ('precision_at_10', 'Precision@10', 3),
        ('recall_at_10', 'Recall@10', 3),
        ('mrr', 'MRR', 3),
        ('coverage', 'Coverage', 3),
        ('diversity', 'Diversity', 3)
    ]

    print("🔍 RETRIEVAL METRICS:")
    print("-" * 80)

    improvements = []
    for metric_key, metric_name, decimals in metrics:
        b_val = baseline['statistics']['retrieval'][f'avg_{metric_key}']
        n_val = new['statistics']['retrieval'][f'avg_{metric_key}']
        diff = n_val - b_val
        pct = (diff / b_val) * 100 if b_val > 0 else 0

        symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➖"
        color = "\033[92m" if diff > 0 else "\033[91m" if diff < 0 else "\033[93m"
        reset = "\033[0m"

        print(f"{symbol} {metric_name:20s}: {b_val:.{decimals}f} → {n_val:.{decimals}f} "
              f"({color}{diff:+.{decimals}f}, {pct:+.1f}%{reset})")

        if diff != 0:
            improvements.append((metric_name, diff, pct))

    # Compare generation quality
    print("\n✨ GENERATION METRICS:")
    print("-" * 80)

    gen_metrics = [
        ('overall_score', 'Overall Score', 3),
        ('completeness', 'Completeness', 3),
        ('formatting_score', 'Formatting', 3)
    ]

    for metric_key, metric_name, decimals in gen_metrics:
        b_val = baseline['statistics']['generation'].get(f'avg_{metric_key}', 0)
        n_val = new['statistics']['generation'].get(f'avg_{metric_key}', 0)
        diff = n_val - b_val
        pct = (diff / b_val) * 100 if b_val > 0 else 0

        symbol = "📈" if diff > 0 else "📉" if diff < 0 else "➖"
        color = "\033[92m" if diff > 0 else "\033[91m" if diff < 0 else "\033[93m"
        reset = "\033[0m"

        print(f"{symbol} {metric_name:20s}: {b_val:.{decimals}f} → {n_val:.{decimals}f} "
              f"({color}{diff:+.{decimals}f}, {pct:+.1f}%{reset})")

    # Compare timing
    print("\n⏱️  PERFORMANCE METRICS:")
    print("-" * 80)

    b_time = baseline['statistics']['avg_query_time']
    n_time = new['statistics']['avg_query_time']
    time_diff = n_time - b_time
    time_pct = (time_diff / b_time) * 100 if b_time > 0 else 0

    symbol = "⚡" if time_diff < 0 else "🐌"
    color = "\033[92m" if time_diff < 0 else "\033[91m"
    reset = "\033[0m"

    print(f"{symbol} Avg Query Time:      {b_time:.2f}s → {n_time:.2f}s "
          f"({color}{time_diff:+.2f}s, {time_pct:+.1f}%{reset})")

    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)

    if improvements:
        print("\n✅ Improvements:")
        for name, diff, pct in improvements:
            if diff > 0:
                print(f"   • {name}: {diff:+.3f} ({pct:+.1f}%)")

        print("\n❌ Regressions:")
        for name, diff, pct in improvements:
            if diff < 0:
                print(f"   • {name}: {diff:+.3f} ({pct:+.1f}%)")

        if time_diff < 0:
            print(f"\n⚡ Speed Improvement: {-time_diff:.2f}s faster ({-time_pct:.1f}%)")
        elif time_diff > 0:
            print(f"\n🐌 Speed Regression: {time_diff:.2f}s slower ({time_pct:.1f}%)")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_evaluations.py <baseline.json> <new.json>")
        sys.exit(1)

    baseline_path = sys.argv[1]
    new_path = sys.argv[2]

    if not Path(baseline_path).exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        sys.exit(1)

    if not Path(new_path).exists():
        print(f"Error: New evaluation file not found: {new_path}")
        sys.exit(1)

    compare_evaluations(baseline_path, new_path)
