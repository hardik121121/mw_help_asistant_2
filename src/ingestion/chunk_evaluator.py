"""
Chunk Quality Evaluation Framework.
Measures how well chunking preserves document structure and semantic coherence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import statistics

from src.ingestion.hierarchical_chunker import HierarchicalChunk

logger = logging.getLogger(__name__)


@dataclass
class ChunkQualityMetrics:
    """Metrics for evaluating chunk quality."""
    # Size metrics
    total_chunks: int
    avg_chunk_size_chars: float
    avg_chunk_size_tokens: float
    std_chunk_size_chars: float
    min_chunk_size_chars: int
    max_chunk_size_chars: int

    # Structure preservation
    chunks_with_headings: int
    avg_heading_depth: float
    sections_preserved: float  # % of chunks that maintain section boundaries

    # Content distribution
    chunks_with_images: int
    chunks_with_tables: int
    chunks_with_code: int
    chunks_with_lists: int

    # Technical depth distribution
    tech_depth_distribution: Dict[str, int]

    # Continuity metrics
    continuation_chunks: int
    avg_chunks_per_section: float

    # Quality scores (0-1)
    size_consistency_score: float  # How consistent are chunk sizes?
    structure_preservation_score: float  # How well is hierarchy preserved?
    context_completeness_score: float  # How complete is contextual metadata?
    overall_quality_score: float


class ChunkQualityEvaluator:
    """Evaluate the quality of hierarchical chunks."""

    def __init__(self, target_chunk_size: int = 1500):
        """
        Initialize evaluator.

        Args:
            target_chunk_size: Expected target chunk size for scoring
        """
        self.target_chunk_size = target_chunk_size

    def evaluate(self, chunks: List[HierarchicalChunk]) -> ChunkQualityMetrics:
        """
        Evaluate chunk quality across multiple dimensions.

        Args:
            chunks: List of hierarchical chunks to evaluate

        Returns:
            ChunkQualityMetrics object with all metrics
        """
        logger.info(f"Evaluating quality of {len(chunks)} chunks...")

        if not chunks:
            logger.error("No chunks to evaluate!")
            return self._empty_metrics()

        # Calculate size metrics
        char_sizes = [c.metadata.char_count for c in chunks]
        token_sizes = [c.metadata.token_count for c in chunks]

        size_metrics = {
            'total_chunks': len(chunks),
            'avg_chunk_size_chars': statistics.mean(char_sizes),
            'avg_chunk_size_tokens': statistics.mean(token_sizes),
            'std_chunk_size_chars': statistics.stdev(char_sizes) if len(char_sizes) > 1 else 0,
            'min_chunk_size_chars': min(char_sizes),
            'max_chunk_size_chars': max(char_sizes)
        }

        # Structure preservation metrics
        chunks_with_headings = sum(1 for c in chunks if c.metadata.current_heading)
        heading_depths = [len(c.metadata.heading_path) for c in chunks if c.metadata.heading_path]
        avg_heading_depth = statistics.mean(heading_depths) if heading_depths else 0

        # Section analysis
        section_ids = [c.metadata.section_id for c in chunks]
        unique_sections = len(set(section_ids))
        chunks_per_section = Counter(section_ids)
        avg_chunks_per_section = statistics.mean(chunks_per_section.values())

        # First chunk of each section is a boundary
        boundary_chunks = sum(1 for c in chunks if c.metadata.chunk_index == 0)
        sections_preserved = (boundary_chunks / unique_sections) if unique_sections > 0 else 0

        structure_metrics = {
            'chunks_with_headings': chunks_with_headings,
            'avg_heading_depth': avg_heading_depth,
            'sections_preserved': sections_preserved
        }

        # Content distribution
        content_metrics = {
            'chunks_with_images': sum(1 for c in chunks if c.metadata.has_images),
            'chunks_with_tables': sum(1 for c in chunks if c.metadata.has_tables),
            'chunks_with_code': sum(1 for c in chunks if c.metadata.has_code),
            'chunks_with_lists': sum(1 for c in chunks if c.metadata.has_lists)
        }

        # Technical depth
        tech_depth_dist = Counter(c.metadata.technical_depth for c in chunks)

        # Continuity
        continuation_chunks = sum(1 for c in chunks if c.metadata.is_continuation)

        # Calculate quality scores
        size_consistency_score = self._calculate_size_consistency_score(char_sizes)
        structure_preservation_score = self._calculate_structure_score(chunks)
        context_completeness_score = self._calculate_context_score(chunks)
        overall_quality_score = (
            size_consistency_score * 0.3 +
            structure_preservation_score * 0.4 +
            context_completeness_score * 0.3
        )

        metrics = ChunkQualityMetrics(
            **size_metrics,
            **structure_metrics,
            **content_metrics,
            tech_depth_distribution=dict(tech_depth_dist),
            continuation_chunks=continuation_chunks,
            avg_chunks_per_section=avg_chunks_per_section,
            size_consistency_score=size_consistency_score,
            structure_preservation_score=structure_preservation_score,
            context_completeness_score=context_completeness_score,
            overall_quality_score=overall_quality_score
        )

        return metrics

    def _calculate_size_consistency_score(self, sizes: List[int]) -> float:
        """
        Calculate how consistent chunk sizes are (0-1 score).

        Penalizes high variance and chunks far from target.
        """
        if not sizes:
            return 0.0

        avg_size = statistics.mean(sizes)
        std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0

        # Coefficient of variation (lower is better)
        cv = std_size / avg_size if avg_size > 0 else 1.0

        # Penalize CV > 0.5 (high variance)
        variance_score = max(0, 1 - cv * 2)

        # How close to target size?
        target_deviation = abs(avg_size - self.target_chunk_size) / self.target_chunk_size
        target_score = max(0, 1 - target_deviation)

        return (variance_score * 0.5 + target_score * 0.5)

    def _calculate_structure_score(self, chunks: List[HierarchicalChunk]) -> float:
        """
        Calculate structure preservation score (0-1).

        Checks:
        - % chunks with heading context
        - % chunks with valid heading paths
        - Heading hierarchy depth consistency
        """
        if not chunks:
            return 0.0

        # Check heading context presence
        with_headings = sum(1 for c in chunks if c.metadata.current_heading or c.metadata.heading_path)
        heading_score = with_headings / len(chunks)

        # Check heading path validity
        with_valid_paths = sum(1 for c in chunks if c.metadata.heading_path)
        path_score = with_valid_paths / len(chunks)

        # Check section ID consistency
        with_section_ids = sum(1 for c in chunks if c.metadata.section_id)
        section_score = with_section_ids / len(chunks)

        return (heading_score * 0.4 + path_score * 0.3 + section_score * 0.3)

    def _calculate_context_score(self, chunks: List[HierarchicalChunk]) -> float:
        """
        Calculate context completeness score (0-1).

        Checks if chunks have rich metadata:
        - Page information
        - Content type classification
        - Technical depth
        - Content features (images, tables, etc.)
        """
        if not chunks:
            return 0.0

        scores = []

        for chunk in chunks:
            chunk_score = 0
            total_checks = 0

            # Page information
            if chunk.metadata.page_start > 0:
                chunk_score += 1
            total_checks += 1

            # Content type
            if chunk.metadata.content_type != "unknown":
                chunk_score += 1
            total_checks += 1

            # Technical depth
            if chunk.metadata.technical_depth in ["low", "medium", "high"]:
                chunk_score += 1
            total_checks += 1

            # Content features detected
            if any([
                chunk.metadata.has_images,
                chunk.metadata.has_tables,
                chunk.metadata.has_code,
                chunk.metadata.has_lists
            ]):
                chunk_score += 1
            total_checks += 1

            # Token count available
            if chunk.metadata.token_count > 0:
                chunk_score += 1
            total_checks += 1

            scores.append(chunk_score / total_checks if total_checks > 0 else 0)

        return statistics.mean(scores) if scores else 0.0

    def _empty_metrics(self) -> ChunkQualityMetrics:
        """Return empty metrics for error cases."""
        return ChunkQualityMetrics(
            total_chunks=0,
            avg_chunk_size_chars=0,
            avg_chunk_size_tokens=0,
            std_chunk_size_chars=0,
            min_chunk_size_chars=0,
            max_chunk_size_chars=0,
            chunks_with_headings=0,
            avg_heading_depth=0,
            sections_preserved=0,
            chunks_with_images=0,
            chunks_with_tables=0,
            chunks_with_code=0,
            chunks_with_lists=0,
            tech_depth_distribution={},
            continuation_chunks=0,
            avg_chunks_per_section=0,
            size_consistency_score=0,
            structure_preservation_score=0,
            context_completeness_score=0,
            overall_quality_score=0
        )

    def analyze_boundaries(self, chunks: List[HierarchicalChunk]) -> Dict:
        """
        Analyze how well chunks respect semantic boundaries.

        Returns:
            Dictionary with boundary analysis
        """
        logger.info("Analyzing semantic boundaries...")

        boundary_analysis = {
            'section_breaks': 0,
            'heading_splits': 0,
            'mid_paragraph_splits': 0,
            'natural_breaks': 0
        }

        for i in range(len(chunks) - 1):
            current = chunks[i]
            next_chunk = chunks[i + 1]

            # Section break (different sections)
            if current.metadata.section_id != next_chunk.metadata.section_id:
                boundary_analysis['section_breaks'] += 1

            # Heading split (different headings)
            elif current.metadata.current_heading != next_chunk.metadata.current_heading:
                boundary_analysis['heading_splits'] += 1

            # Check if break is at natural boundary
            elif current.content.rstrip().endswith(('.', '!', '?', '\n')):
                boundary_analysis['natural_breaks'] += 1

            else:
                boundary_analysis['mid_paragraph_splits'] += 1

        total_breaks = len(chunks) - 1
        if total_breaks > 0:
            boundary_analysis['natural_break_ratio'] = (
                (boundary_analysis['natural_breaks'] + boundary_analysis['section_breaks']) /
                total_breaks
            )
        else:
            boundary_analysis['natural_break_ratio'] = 0

        return boundary_analysis

    def find_problematic_chunks(self, chunks: List[HierarchicalChunk]) -> Dict:
        """
        Identify chunks that may cause retrieval issues.

        Returns:
            Dictionary with lists of problematic chunks
        """
        logger.info("Identifying problematic chunks...")

        problems = {
            'too_small': [],
            'too_large': [],
            'missing_context': [],
            'missing_heading': [],
            'isolated_chunks': []
        }

        for chunk in chunks:
            chunk_id = chunk.metadata.chunk_id

            # Too small (less than 1/3 of target)
            if chunk.metadata.char_count < self.target_chunk_size / 3:
                problems['too_small'].append({
                    'chunk_id': chunk_id,
                    'size': chunk.metadata.char_count,
                    'page': chunk.metadata.page_start
                })

            # Too large (more than 2x target)
            if chunk.metadata.char_count > self.target_chunk_size * 2:
                problems['too_large'].append({
                    'chunk_id': chunk_id,
                    'size': chunk.metadata.char_count,
                    'page': chunk.metadata.page_start
                })

            # Missing contextual heading path
            if not chunk.metadata.heading_path:
                problems['missing_context'].append({
                    'chunk_id': chunk_id,
                    'page': chunk.metadata.page_start
                })

            # No heading information at all
            if not chunk.metadata.current_heading and not chunk.metadata.heading_path:
                problems['missing_heading'].append({
                    'chunk_id': chunk_id,
                    'page': chunk.metadata.page_start
                })

            # Isolated chunk (section has only 1 chunk and it's very small)
            if (chunk.metadata.total_chunks_in_section == 1 and
                chunk.metadata.char_count < self.target_chunk_size / 2):
                problems['isolated_chunks'].append({
                    'chunk_id': chunk_id,
                    'size': chunk.metadata.char_count,
                    'page': chunk.metadata.page_start
                })

        return problems

    def generate_report(
        self,
        chunks: List[HierarchicalChunk],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive quality report.

        Args:
            chunks: List of chunks to evaluate
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        logger.info("Generating chunk quality report...")

        metrics = self.evaluate(chunks)
        boundaries = self.analyze_boundaries(chunks)
        problems = self.find_problematic_chunks(chunks)

        # Build report
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("CHUNK QUALITY EVALUATION REPORT")
        report_lines.append("=" * 70)

        # Overview
        report_lines.append(f"\n📊 OVERVIEW")
        report_lines.append(f"   Total Chunks: {metrics.total_chunks}")
        report_lines.append(f"   Overall Quality Score: {metrics.overall_quality_score:.2f}/1.00")

        # Size metrics
        report_lines.append(f"\n📏 SIZE METRICS")
        report_lines.append(f"   Average Size: {metrics.avg_chunk_size_chars:.0f} chars ({metrics.avg_chunk_size_tokens:.0f} tokens)")
        report_lines.append(f"   Size Range: {metrics.min_chunk_size_chars} - {metrics.max_chunk_size_chars} chars")
        report_lines.append(f"   Standard Deviation: {metrics.std_chunk_size_chars:.0f} chars")
        report_lines.append(f"   Size Consistency Score: {metrics.size_consistency_score:.2f}/1.00")

        # Structure metrics
        report_lines.append(f"\n🏗️  STRUCTURE PRESERVATION")
        report_lines.append(f"   Chunks with Headings: {metrics.chunks_with_headings} ({metrics.chunks_with_headings/metrics.total_chunks*100:.1f}%)")
        report_lines.append(f"   Average Heading Depth: {metrics.avg_heading_depth:.1f}")
        report_lines.append(f"   Sections Preserved: {metrics.sections_preserved*100:.1f}%")
        report_lines.append(f"   Avg Chunks per Section: {metrics.avg_chunks_per_section:.1f}")
        report_lines.append(f"   Structure Preservation Score: {metrics.structure_preservation_score:.2f}/1.00")

        # Content distribution
        report_lines.append(f"\n📦 CONTENT DISTRIBUTION")
        report_lines.append(f"   With Images: {metrics.chunks_with_images} ({metrics.chunks_with_images/metrics.total_chunks*100:.1f}%)")
        report_lines.append(f"   With Tables: {metrics.chunks_with_tables} ({metrics.chunks_with_tables/metrics.total_chunks*100:.1f}%)")
        report_lines.append(f"   With Code: {metrics.chunks_with_code} ({metrics.chunks_with_code/metrics.total_chunks*100:.1f}%)")
        report_lines.append(f"   With Lists: {metrics.chunks_with_lists} ({metrics.chunks_with_lists/metrics.total_chunks*100:.1f}%)")

        # Technical depth
        report_lines.append(f"\n🔬 TECHNICAL DEPTH DISTRIBUTION")
        for depth, count in sorted(metrics.tech_depth_distribution.items()):
            pct = count / metrics.total_chunks * 100
            report_lines.append(f"   {depth.capitalize()}: {count} ({pct:.1f}%)")

        # Boundary analysis
        report_lines.append(f"\n🔗 BOUNDARY ANALYSIS")
        report_lines.append(f"   Section Breaks: {boundaries['section_breaks']}")
        report_lines.append(f"   Heading Splits: {boundaries['heading_splits']}")
        report_lines.append(f"   Natural Breaks: {boundaries['natural_breaks']}")
        report_lines.append(f"   Mid-paragraph Splits: {boundaries['mid_paragraph_splits']}")
        report_lines.append(f"   Natural Break Ratio: {boundaries['natural_break_ratio']:.2f}")

        # Problems
        report_lines.append(f"\n⚠️  POTENTIAL ISSUES")
        report_lines.append(f"   Too Small: {len(problems['too_small'])} chunks")
        report_lines.append(f"   Too Large: {len(problems['too_large'])} chunks")
        report_lines.append(f"   Missing Context: {len(problems['missing_context'])} chunks")
        report_lines.append(f"   Missing Heading: {len(problems['missing_heading'])} chunks")
        report_lines.append(f"   Isolated Chunks: {len(problems['isolated_chunks'])} chunks")

        # Quality scores summary
        report_lines.append(f"\n✨ QUALITY SCORES")
        report_lines.append(f"   Size Consistency: {metrics.size_consistency_score:.2f}/1.00")
        report_lines.append(f"   Structure Preservation: {metrics.structure_preservation_score:.2f}/1.00")
        report_lines.append(f"   Context Completeness: {metrics.context_completeness_score:.2f}/1.00")
        report_lines.append(f"   OVERALL QUALITY: {metrics.overall_quality_score:.2f}/1.00")

        # Recommendations
        report_lines.append(f"\n💡 RECOMMENDATIONS")
        if metrics.size_consistency_score < 0.7:
            report_lines.append("   - Consider adjusting chunk_size or chunk_overlap for better consistency")
        if metrics.structure_preservation_score < 0.8:
            report_lines.append("   - Review heading detection logic to improve structure preservation")
        if len(problems['too_small']) > metrics.total_chunks * 0.1:
            report_lines.append("   - Many chunks are too small; consider increasing min_chunk_size")
        if len(problems['missing_heading']) > 0:
            report_lines.append("   - Some chunks lack heading context; review document structure")

        if metrics.overall_quality_score >= 0.8:
            report_lines.append("   ✅ Chunk quality is excellent! No major issues detected.")
        elif metrics.overall_quality_score >= 0.6:
            report_lines.append("   ⚠️  Chunk quality is acceptable but could be improved.")
        else:
            report_lines.append("   ❌ Chunk quality needs improvement. Review settings and document structure.")

        report_lines.append("\n" + "=" * 70)

        report = "\n".join(report_lines)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            # Also save metrics as JSON
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics': asdict(metrics),
                    'boundaries': boundaries,
                    'problems': problems
                }, f, indent=2)

            logger.info(f"Report saved to: {output_path}")
            logger.info(f"Metrics saved to: {json_path}")

        return report


def main():
    """Example usage of ChunkQualityEvaluator."""
    from src.ingestion.hierarchical_chunker import HierarchicalChunker

    # Load chunks
    chunks_path = "cache/hierarchical_chunks.json"
    print(f"Loading chunks from: {chunks_path}")

    chunks = HierarchicalChunker.load_from_json(chunks_path)

    # Evaluate
    evaluator = ChunkQualityEvaluator(target_chunk_size=1500)

    report = evaluator.generate_report(
        chunks,
        output_path="tests/results/chunk_quality_report.txt"
    )

    print(report)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
