"""
TOC (Table of Contents) Filtering Utility.
Filters out or marks TOC chunks for better retrieval quality.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class TOCFilter:
    """
    Filters or marks Table of Contents chunks in the document.

    The first 18 pages of the PDF are the index/TOC. This utility helps:
    1. Filter them out entirely for retrieval
    2. Mark them with metadata for special handling
    3. Keep them but with lower priority
    """

    def __init__(self, toc_end_page: int = 18):
        """
        Initialize TOC filter.

        Args:
            toc_end_page: Last page number of TOC (default: 18)
        """
        self.toc_end_page = toc_end_page
        logger.info(f"Initialized TOCFilter: TOC ends at page {toc_end_page}")

    def filter_chunks(self, chunks: List[Dict],
                     strategy: str = "mark") -> List[Dict]:
        """
        Filter or mark TOC chunks.

        Args:
            chunks: List of chunk dictionaries
            strategy: "remove" | "mark" | "deprioritize"
                - remove: Completely remove TOC chunks
                - mark: Add is_toc flag to metadata
                - deprioritize: Add lower technical_depth to TOC chunks

        Returns:
            Filtered/marked chunk list
        """
        if strategy == "remove":
            return self._remove_toc_chunks(chunks)
        elif strategy == "mark":
            return self._mark_toc_chunks(chunks)
        elif strategy == "deprioritize":
            return self._deprioritize_toc_chunks(chunks)
        else:
            logger.warning(f"Unknown strategy: {strategy}, returning original chunks")
            return chunks

    def _remove_toc_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove TOC chunks entirely."""
        original_count = len(chunks)
        filtered = [
            c for c in chunks
            if c['metadata']['page_start'] > self.toc_end_page
        ]
        removed_count = original_count - len(filtered)

        logger.info(f"Removed {removed_count} TOC chunks ({removed_count/original_count*100:.1f}%)")
        logger.info(f"Remaining chunks: {len(filtered)}")

        return filtered

    def _mark_toc_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add is_toc flag to metadata."""
        toc_count = 0

        for chunk in chunks:
            is_toc = chunk['metadata']['page_start'] <= self.toc_end_page
            chunk['metadata']['is_toc'] = is_toc

            if is_toc:
                toc_count += 1
                # Also mark content type
                chunk['metadata']['content_type'] = 'table_of_contents'

        logger.info(f"Marked {toc_count} TOC chunks ({toc_count/len(chunks)*100:.1f}%)")

        return chunks

    def _deprioritize_toc_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Lower priority of TOC chunks."""
        toc_count = 0

        for chunk in chunks:
            if chunk['metadata']['page_start'] <= self.toc_end_page:
                chunk['metadata']['is_toc'] = True
                chunk['metadata']['technical_depth'] = 'low'
                chunk['metadata']['content_type'] = 'table_of_contents'
                toc_count += 1

        logger.info(f"Deprioritized {toc_count} TOC chunks")

        return chunks

    def get_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about TOC vs content chunks."""
        toc_chunks = [
            c for c in chunks
            if c['metadata']['page_start'] <= self.toc_end_page
        ]
        content_chunks = [
            c for c in chunks
            if c['metadata']['page_start'] > self.toc_end_page
        ]

        return {
            'total_chunks': len(chunks),
            'toc_chunks': len(toc_chunks),
            'content_chunks': len(content_chunks),
            'toc_percentage': len(toc_chunks) / len(chunks) * 100,
            'toc_pages': f"1-{self.toc_end_page}",
            'content_pages': f"{self.toc_end_page + 1}+",
            'toc_page_range': list(range(1, self.toc_end_page + 1)),
        }


def filter_chunks_file(input_path: str, output_path: str,
                      strategy: str = "mark", toc_end_page: int = 18):
    """
    Filter chunks from file and save to new file.

    Args:
        input_path: Path to input chunks JSON
        output_path: Path to output filtered chunks JSON
        strategy: Filter strategy (remove, mark, deprioritize)
        toc_end_page: Last page of TOC
    """
    print(f"\n{'='*60}")
    print("🔍 TOC FILTERING")
    print(f"{'='*60}\n")

    # Load chunks
    print(f"Loading chunks from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data.get('chunks', [])
    print(f"Loaded {len(chunks)} chunks\n")

    # Initialize filter
    toc_filter = TOCFilter(toc_end_page=toc_end_page)

    # Get stats before filtering
    print("Before filtering:")
    stats_before = toc_filter.get_stats(chunks)
    print(f"  Total chunks: {stats_before['total_chunks']}")
    print(f"  TOC chunks (pages {stats_before['toc_pages']}): {stats_before['toc_chunks']} ({stats_before['toc_percentage']:.1f}%)")
    print(f"  Content chunks (pages {stats_before['content_pages']}): {stats_before['content_chunks']}")

    # Apply filter
    print(f"\nApplying strategy: '{strategy}'...")
    filtered_chunks = toc_filter.filter_chunks(chunks, strategy=strategy)

    # Get stats after filtering
    print("\nAfter filtering:")
    stats_after = toc_filter.get_stats(filtered_chunks)
    print(f"  Total chunks: {stats_after['total_chunks']}")
    print(f"  TOC chunks: {stats_after['toc_chunks']}")
    print(f"  Content chunks: {stats_after['content_chunks']}")

    # Save filtered chunks
    data['chunks'] = filtered_chunks
    data['metadata'] = data.get('metadata', {})
    data['metadata']['toc_filtered'] = True
    data['metadata']['toc_strategy'] = strategy
    data['metadata']['toc_end_page'] = toc_end_page

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved filtered chunks to: {output_path}")
    print(f"{'='*60}\n")


def main():
    """Test TOC filtering with different strategies."""
    import argparse

    parser = argparse.ArgumentParser(description='Filter TOC chunks from hierarchical chunks')
    parser.add_argument('--input', default='cache/hierarchical_chunks.json',
                       help='Input chunks JSON file')
    parser.add_argument('--output', default='cache/hierarchical_chunks_filtered.json',
                       help='Output chunks JSON file')
    parser.add_argument('--strategy', choices=['remove', 'mark', 'deprioritize'],
                       default='mark',
                       help='Filtering strategy')
    parser.add_argument('--toc-end-page', type=int, default=18,
                       help='Last page number of TOC')

    args = parser.parse_args()

    filter_chunks_file(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        toc_end_page=args.toc_end_page
    )


if __name__ == "__main__":
    main()
