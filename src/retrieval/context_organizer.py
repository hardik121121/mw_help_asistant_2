"""
Context Organizer for aggregating and organizing retrieval results.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrganizedContext:
    """
    Organized retrieval context for generation.

    Attributes:
        chunks: Deduplicated and ordered chunks
        topic_groups: Chunks grouped by topic
        section_hierarchy: Section organization
        total_chunks: Total number of chunks
        unique_sections: Number of unique sections
        page_range: (min_page, max_page)
        has_images: Whether context includes images
        has_tables: Whether context includes tables
        has_code: Whether context includes code examples
    """
    chunks: List[Dict] = field(default_factory=list)
    topic_groups: Dict[str, List[Dict]] = field(default_factory=dict)
    section_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    total_chunks: int = 0
    unique_sections: int = 0
    page_range: Tuple[int, int] = (0, 0)
    has_images: bool = False
    has_tables: bool = False
    has_code: bool = False


class ContextOrganizer:
    """
    Organizes and aggregates retrieval results for generation.

    Features:
    - Deduplication
    - Topic clustering
    - Hierarchical organization
    - Content type aggregation
    - Relationship mapping
    """

    def __init__(self):
        """Initialize context organizer."""
        logger.info("Initialized ContextOrganizer")

    def organize(self,
                results_by_subquestion: List[List[Dict]],
                max_chunks: int = 20,
                preserve_order: bool = True) -> OrganizedContext:
        """
        Organize retrieval results from multiple sub-questions.

        Args:
            results_by_subquestion: List of result lists (one per sub-question)
            max_chunks: Maximum chunks to include in final context
            preserve_order: Preserve sub-question ordering

        Returns:
            OrganizedContext with aggregated results
        """
        logger.info(f"Organizing results from {len(results_by_subquestion)} sub-questions")

        # 1. Deduplicate across sub-questions
        deduplicated = self._deduplicate(results_by_subquestion, preserve_order)
        logger.info(f"  Deduplicated: {len(deduplicated)} unique chunks")

        # 2. Limit to max_chunks (keep highest scores)
        limited = self._limit_chunks(deduplicated, max_chunks)
        logger.info(f"  Limited to: {len(limited)} chunks")

        # 3. Group by topic
        topic_groups = self._group_by_topic(limited)
        logger.info(f"  Topic groups: {len(topic_groups)}")

        # 4. Build section hierarchy
        section_hierarchy = self._build_section_hierarchy(limited)
        logger.info(f"  Section hierarchy: {len(section_hierarchy)} top-level sections")

        # 5. Sort chunks for optimal reading flow
        sorted_chunks = self._sort_for_readability(limited)

        # 6. Extract metadata
        metadata = self._extract_metadata(sorted_chunks)

        # Build organized context
        context = OrganizedContext(
            chunks=sorted_chunks,
            topic_groups=topic_groups,
            section_hierarchy=section_hierarchy,
            total_chunks=len(sorted_chunks),
            unique_sections=len(section_hierarchy),
            page_range=metadata['page_range'],
            has_images=metadata['has_images'],
            has_tables=metadata['has_tables'],
            has_code=metadata['has_code']
        )

        logger.info("✅ Context organization complete")
        return context

    def _deduplicate(self,
                    results_by_subquestion: List[List[Dict]],
                    preserve_order: bool = True) -> List[Dict]:
        """
        Deduplicate chunks across sub-questions.

        Args:
            results_by_subquestion: Results lists
            preserve_order: Keep first occurrence order

        Returns:
            Deduplicated chunk list
        """
        seen_ids = set()
        deduplicated = []
        chunk_scores = defaultdict(list)

        # Collect all chunks with their scores
        for subq_idx, results in enumerate(results_by_subquestion):
            for result in results:
                chunk_id = result['chunk_id']
                chunk_scores[chunk_id].append(result.get('score', 0))

                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    # Store which sub-question(s) this chunk is relevant to
                    if 'relevant_subquestions' not in result:
                        result['relevant_subquestions'] = []
                    result['relevant_subquestions'].append(subq_idx)
                    deduplicated.append(result)

        # Update scores with average across sub-questions
        for chunk in deduplicated:
            chunk_id = chunk['chunk_id']
            scores = chunk_scores[chunk_id]
            chunk['aggregated_score'] = sum(scores) / len(scores)
            chunk['num_subquestions'] = len(scores)

        return deduplicated

    def _limit_chunks(self, chunks: List[Dict], max_chunks: int) -> List[Dict]:
        """
        Limit to max_chunks, keeping highest scoring chunks.

        Args:
            chunks: Chunk list
            max_chunks: Maximum number to keep

        Returns:
            Limited chunk list
        """
        if len(chunks) <= max_chunks:
            return chunks

        # Sort by aggregated score (or original score)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('aggregated_score', x.get('score', 0)),
            reverse=True
        )

        return sorted_chunks[:max_chunks]

    def _group_by_topic(self, chunks: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group chunks by topic (top-level heading).

        Args:
            chunks: Chunk list

        Returns:
            Dict mapping topic -> chunk list
        """
        topic_groups = defaultdict(list)

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])

            # Use first heading as topic
            topic = heading_path[0] if heading_path else "General"
            topic_groups[topic].append(chunk)

        return dict(topic_groups)

    def _build_section_hierarchy(self, chunks: List[Dict]) -> Dict[str, List[str]]:
        """
        Build hierarchical section structure.

        Args:
            chunks: Chunk list

        Returns:
            Dict mapping parent section -> child sections
        """
        hierarchy = defaultdict(set)

        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])

            # Build parent-child relationships
            for i in range(len(heading_path) - 1):
                parent = heading_path[i]
                child = heading_path[i + 1]
                hierarchy[parent].add(child)

        # Convert sets to sorted lists
        return {k: sorted(list(v)) for k, v in hierarchy.items()}

    def _sort_for_readability(self, chunks: List[Dict]) -> List[Dict]:
        """
        Sort chunks for optimal reading flow.

        Strategy:
        1. Group by top-level section
        2. Within section, sort by page number
        3. Within page, sort by heading level

        Args:
            chunks: Chunk list

        Returns:
            Sorted chunk list
        """
        def sort_key(chunk):
            metadata = chunk.get('metadata', {})
            heading_path = metadata.get('heading_path', [])
            page_start = metadata.get('page_start', 0)
            heading_level = metadata.get('heading_level', 0)

            # Primary: top-level section
            section = heading_path[0] if heading_path else ""

            # Secondary: page number
            # Tertiary: heading level (shallower first)
            return (section, page_start, heading_level)

        sorted_chunks = sorted(chunks, key=sort_key)
        return sorted_chunks

    def _extract_metadata(self, chunks: List[Dict]) -> Dict:
        """
        Extract aggregate metadata from chunks.

        Args:
            chunks: Chunk list

        Returns:
            Metadata dict
        """
        pages = []
        has_images = False
        has_tables = False
        has_code = False

        for chunk in chunks:
            metadata = chunk.get('metadata', {})

            # Collect pages
            page_start = metadata.get('page_start')
            page_end = metadata.get('page_end')
            if page_start:
                pages.append(page_start)
            if page_end:
                pages.append(page_end)

            # Check content types
            if metadata.get('has_images'):
                has_images = True
            if metadata.get('has_tables'):
                has_tables = True
            if metadata.get('has_code'):
                has_code = True

        # Calculate page range
        page_range = (min(pages), max(pages)) if pages else (0, 0)

        return {
            'page_range': page_range,
            'has_images': has_images,
            'has_tables': has_tables,
            'has_code': has_code
        }

    def format_for_generation(self, context: OrganizedContext) -> str:
        """
        Format organized context as text for LLM generation.

        Args:
            context: OrganizedContext

        Returns:
            Formatted context string
        """
        sections = []

        # Header
        sections.append("=" * 70)
        sections.append("RETRIEVED CONTEXT")
        sections.append("=" * 70)
        sections.append(f"Total Chunks: {context.total_chunks}")
        sections.append(f"Sections: {context.unique_sections}")
        sections.append(f"Page Range: {context.page_range[0]}-{context.page_range[1]}")
        sections.append("")

        # Chunks by topic
        for topic, chunks in context.topic_groups.items():
            sections.append(f"\n## {topic}")
            sections.append(f"({len(chunks)} chunks)")
            sections.append("")

            for i, chunk in enumerate(chunks, 1):
                metadata = chunk.get('metadata', {})
                heading_path = metadata.get('heading_path', [])
                page_start = metadata.get('page_start', 0)

                sections.append(f"### Chunk {i}: {' > '.join(heading_path)}")
                sections.append(f"(Page {page_start})")
                sections.append("")
                sections.append(chunk['content'])
                sections.append("")

        return "\n".join(sections)


if __name__ == "__main__":
    """Test context organizer."""
    print("\n" + "="*70)
    print("TESTING CONTEXT ORGANIZER")
    print("="*70 + "\n")

    # Initialize
    organizer = ContextOrganizer()

    # Simulate multi-step retrieval results
    from src.retrieval.hybrid_search import HybridSearch
    from src.database.embedding_generator import EmbeddingGenerator

    hybrid_search = HybridSearch()
    generator = EmbeddingGenerator()

    # Test queries (simulating decomposed query)
    test_queries = [
        "What are no-code blocks in Watermelon?",
        "How to create a no-code block step by step?",
        "What is Autonomous Functional Testing?"
    ]

    print(f"Simulating retrieval for {len(test_queries)} sub-questions:\n")
    for i, q in enumerate(test_queries, 1):
        print(f"{i}. {q}")
    print()

    # Get results for each sub-question
    results_by_subq = []
    for query in test_queries:
        query_embedding = generator.generate_embeddings([query])[0]
        results = hybrid_search.search(
            query=query,
            query_embedding=query_embedding,
            top_k=10
        )
        results_by_subq.append(results)
        print(f"  Sub-Q {len(results_by_subq)}: {len(results)} results")

    # Organize
    print("\nOrganizing context...")
    context = organizer.organize(results_by_subq, max_chunks=15)

    # Display organized context
    print(f"\n📊 Organized Context:\n")
    print(f"Total Chunks: {context.total_chunks}")
    print(f"Unique Sections: {context.unique_sections}")
    print(f"Page Range: {context.page_range}")
    print(f"Has Images: {context.has_images}")
    print(f"Has Tables: {context.has_tables}")
    print(f"Has Code: {context.has_code}")

    print(f"\nTopic Groups ({len(context.topic_groups)}):")
    for topic, chunks in context.topic_groups.items():
        print(f"  - {topic}: {len(chunks)} chunks")

    print(f"\nSection Hierarchy:")
    for parent, children in list(context.section_hierarchy.items())[:5]:
        print(f"  {parent}:")
        for child in children[:3]:
            print(f"    - {child}")

    print("\n✅ Context organizer test complete!")
