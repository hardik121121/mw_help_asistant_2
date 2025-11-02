"""
Hierarchical Chunking System for RAG.
Preserves document structure, merges multi-page topics, and adds contextual metadata.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import tiktoken
except ImportError:
    print("⚠️  LangChain or tiktoken not installed.")
    print("Please run: pip install langchain-text-splitters tiktoken")
    RecursiveCharacterTextSplitter = None
    tiktoken = None

from src.ingestion.docling_processor import DocumentStructure, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk."""
    # Location information
    chunk_id: str
    page_start: int
    page_end: int
    section_id: str

    # Hierarchical context
    heading_path: List[str]  # Full path: ["Chapter 1", "Section 1.1", ...]
    current_heading: Optional[str] = None
    heading_level: Optional[int] = None

    # Content characteristics
    content_type: str = "mixed"  # text, table, list, code, mixed
    technical_depth: str = "medium"  # low, medium, high
    has_images: bool = False
    has_tables: bool = False
    has_code: bool = False
    has_lists: bool = False

    # Image references
    image_paths: List[str] = field(default_factory=list)
    image_captions: List[str] = field(default_factory=list)

    # Table references
    table_texts: List[str] = field(default_factory=list)

    # Chunking metadata
    is_continuation: bool = False  # True if chunk is part of split section
    chunk_index: int = 0  # Index within the section
    total_chunks_in_section: int = 1

    # Token information
    token_count: int = 0
    char_count: int = 0


@dataclass
class HierarchicalChunk:
    """A chunk with hierarchical context and rich metadata."""
    content: str  # The actual text content
    metadata: ChunkMetadata


class HierarchicalChunker:
    """
    Advanced chunking system that preserves document hierarchy and context.

    Key Features:
    1. Section-based chunking (respects heading boundaries)
    2. Context injection (prepends section hierarchy to each chunk)
    3. Multi-page topic handling (merges content that spans pages)
    4. Rich metadata for each chunk
    5. Smart chunk sizing with overlap
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        min_chunk_size: int = 200,
        encoding_name: str = "cl100k_base"  # OpenAI's tokenizer
    ):
        """
        Initialize hierarchical chunker.

        Args:
            chunk_size: Target size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to create
            encoding_name: Tokenizer encoding name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding_name = encoding_name

        if RecursiveCharacterTextSplitter is None or tiktoken is None:
            raise ImportError(
                "LangChain and tiktoken are required. "
                "Please run: pip install langchain-text-splitters tiktoken"
            )

        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using character count.")
            self.tokenizer = None

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # Character-based
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

        logger.info(
            f"Initialized HierarchicalChunker: "
            f"size={chunk_size}, overlap={chunk_overlap}"
        )

    def chunk_document(
        self,
        structure: DocumentStructure,
        include_images: bool = True,
        include_tables: bool = True
    ) -> List[HierarchicalChunk]:
        """
        Chunk the document while preserving hierarchy and context.

        Args:
            structure: DocumentStructure from Docling processor
            include_images: Whether to include image references in metadata
            include_tables: Whether to include table content in chunks

        Returns:
            List of HierarchicalChunk objects
        """
        logger.info("Starting hierarchical chunking...")

        # Group content by sections
        sections = self._group_by_sections(structure.text_blocks)

        logger.info(f"Grouped content into {len(sections)} sections")

        # Build image and table lookups by page
        image_lookup = self._build_image_lookup(structure.images) if include_images else {}
        table_lookup = self._build_table_lookup(structure.tables) if include_tables else {}

        # Chunk each section
        all_chunks = []
        for section_idx, section in enumerate(sections, 1):
            section_chunks = self._chunk_section(
                section,
                section_idx,
                image_lookup,
                table_lookup
            )
            all_chunks.extend(section_chunks)

        logger.info(f"Created {len(all_chunks)} hierarchical chunks")

        # Validate chunks
        self._validate_chunks(all_chunks)

        return all_chunks

    def _group_by_sections(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """
        Group text blocks into logical sections based on headings.

        Each section includes:
        - Heading (if present)
        - All content until next heading of same or higher level
        - Hierarchical context

        Returns:
            List of section dictionaries
        """
        sections = []
        current_section = None

        for block in text_blocks:
            if block.block_type == 'heading':
                # Save previous section
                if current_section and current_section['content']:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'heading': block.text,
                    'level': block.level,
                    'heading_path': block.heading_path + [block.text],
                    'page_start': block.page_number,
                    'page_end': block.page_number,
                    'content': [block.text],  # Include heading in content
                    'blocks': [block],
                    'content_types': set(['heading'])
                }

            elif current_section is not None:
                # Add to current section
                current_section['content'].append(block.text)
                current_section['blocks'].append(block)
                current_section['page_end'] = block.page_number
                current_section['content_types'].add(block.block_type)

            else:
                # No heading yet, create default section
                current_section = {
                    'heading': None,
                    'level': None,
                    'heading_path': [],
                    'page_start': block.page_number,
                    'page_end': block.page_number,
                    'content': [block.text],
                    'blocks': [block],
                    'content_types': set([block.block_type])
                }

        # Add last section
        if current_section and current_section['content']:
            sections.append(current_section)

        return sections

    def _chunk_section(
        self,
        section: Dict,
        section_idx: int,
        image_lookup: Dict[int, List],
        table_lookup: Dict[int, List]
    ) -> List[HierarchicalChunk]:
        """
        Chunk a single section, potentially splitting if too long.

        Args:
            section: Section dictionary
            section_idx: Section index for ID generation
            image_lookup: Page -> images mapping
            table_lookup: Page -> tables mapping

        Returns:
            List of chunks for this section
        """
        # Build section text with context
        heading_path = section['heading_path']
        section_text = "\n\n".join(section['content'])

        # Add context header
        if heading_path:
            context_header = " > ".join(heading_path)
            full_text = f"Section: {context_header}\n\n{section_text}"
        else:
            full_text = section_text

        # Collect images and tables in this section
        page_range = range(section['page_start'], section['page_end'] + 1)
        section_images = []
        section_tables = []

        for page in page_range:
            section_images.extend(image_lookup.get(page, []))
            section_tables.extend(table_lookup.get(page, []))

        # Determine content type
        content_type = self._classify_content_type(section['content_types'])

        # Determine technical depth
        technical_depth = self._estimate_technical_depth(section_text)

        # Check if section needs splitting
        if len(full_text) <= self.chunk_size:
            # Single chunk for this section
            chunk = self._create_chunk(
                content=full_text,
                section_id=f"sec_{section_idx}",
                chunk_index=0,
                total_chunks=1,
                heading_path=heading_path,
                current_heading=section['heading'],
                heading_level=section['level'],
                page_start=section['page_start'],
                page_end=section['page_end'],
                content_type=content_type,
                technical_depth=technical_depth,
                images=section_images,
                tables=section_tables,
                is_continuation=False
            )
            return [chunk]

        else:
            # Split section into multiple chunks
            return self._split_long_section(
                full_text=full_text,
                section=section,
                section_idx=section_idx,
                content_type=content_type,
                technical_depth=technical_depth,
                images=section_images,
                tables=section_tables
            )

    def _split_long_section(
        self,
        full_text: str,
        section: Dict,
        section_idx: int,
        content_type: str,
        technical_depth: str,
        images: List,
        tables: List
    ) -> List[HierarchicalChunk]:
        """Split a long section into multiple overlapping chunks."""
        # Use RecursiveCharacterTextSplitter for intelligent splitting
        text_chunks = self.text_splitter.split_text(full_text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Distribute images across chunks (first chunk gets most)
            chunk_images = images[:3] if i == 0 else images[3:5] if i == 1 else []

            # Add first table to first chunk, others to subsequent chunks
            chunk_tables = [tables[i]] if i < len(tables) else []

            chunk = self._create_chunk(
                content=chunk_text,
                section_id=f"sec_{section_idx}",
                chunk_index=i,
                total_chunks=len(text_chunks),
                heading_path=section['heading_path'],
                current_heading=section['heading'],
                heading_level=section['level'],
                page_start=section['page_start'],
                page_end=section['page_end'],
                content_type=content_type,
                technical_depth=technical_depth,
                images=chunk_images,
                tables=chunk_tables,
                is_continuation=(i > 0)
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        content: str,
        section_id: str,
        chunk_index: int,
        total_chunks: int,
        heading_path: List[str],
        current_heading: Optional[str],
        heading_level: Optional[int],
        page_start: int,
        page_end: int,
        content_type: str,
        technical_depth: str,
        images: List,
        tables: List,
        is_continuation: bool
    ) -> HierarchicalChunk:
        """Create a single HierarchicalChunk with complete metadata."""

        # Generate unique chunk ID
        chunk_id = f"{section_id}_chunk_{chunk_index}"

        # Extract image information
        image_paths = [img.file_path for img in images]
        image_captions = [img.caption for img in images if img.caption]

        # Extract table information
        table_texts = [table.text for table in tables if table.text]

        # Count tokens
        token_count = self._count_tokens(content)
        char_count = len(content)

        # Detect content features
        has_images = len(images) > 0
        has_tables = len(tables) > 0
        has_code = self._contains_code(content)
        has_lists = self._contains_lists(content)

        # Create metadata
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            page_start=page_start,
            page_end=page_end,
            section_id=section_id,
            heading_path=heading_path,
            current_heading=current_heading,
            heading_level=heading_level,
            content_type=content_type,
            technical_depth=technical_depth,
            has_images=has_images,
            has_tables=has_tables,
            has_code=has_code,
            has_lists=has_lists,
            image_paths=image_paths,
            image_captions=image_captions,
            table_texts=table_texts,
            is_continuation=is_continuation,
            chunk_index=chunk_index,
            total_chunks_in_section=total_chunks,
            token_count=token_count,
            char_count=char_count
        )

        return HierarchicalChunk(content=content, metadata=metadata)

    def _build_image_lookup(self, images: List) -> Dict[int, List]:
        """Build page number -> images mapping."""
        lookup = defaultdict(list)
        for img in images:
            lookup[img.page_number].append(img)
        return dict(lookup)

    def _build_table_lookup(self, tables: List) -> Dict[int, List]:
        """Build page number -> tables mapping."""
        lookup = defaultdict(list)
        for table in tables:
            lookup[table.page_number].append(table)
        return dict(lookup)

    def _classify_content_type(self, content_types: set) -> str:
        """Classify content type based on block types."""
        if len(content_types) > 2:
            return "mixed"
        elif 'table' in content_types:
            return "table"
        elif 'code' in content_types:
            return "code"
        elif 'list_item' in content_types:
            return "list"
        else:
            return "text"

    def _estimate_technical_depth(self, text: str) -> str:
        """Estimate technical complexity of content."""
        # Simple heuristic based on technical keywords
        technical_keywords = [
            'api', 'endpoint', 'configuration', 'authentication', 'webhook',
            'integration', 'parameter', 'json', 'xml', 'http', 'request',
            'response', 'error', 'exception', 'function', 'method', 'class'
        ]

        text_lower = text.lower()
        tech_count = sum(1 for keyword in technical_keywords if keyword in text_lower)

        if tech_count >= 5:
            return "high"
        elif tech_count >= 2:
            return "medium"
        else:
            return "low"

    def _contains_code(self, text: str) -> bool:
        """Check if text contains code blocks."""
        code_indicators = ['```', '`', '    ', 'def ', 'function ', 'class ', '{', '}']
        return any(indicator in text for indicator in code_indicators)

    def _contains_lists(self, text: str) -> bool:
        """Check if text contains lists."""
        list_indicators = ['\n- ', '\n* ', '\n1. ', '\n2. ', '\n•']
        return any(indicator in text for indicator in list_indicators)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Fallback: estimate tokens as words / 0.75
        return int(len(text.split()) / 0.75)

    def _validate_chunks(self, chunks: List[HierarchicalChunk]):
        """Validate chunk quality."""
        if not chunks:
            logger.warning("No chunks created!")
            return

        # Check for very small chunks
        small_chunks = [c for c in chunks if c.metadata.char_count < self.min_chunk_size]
        if small_chunks:
            logger.warning(f"Found {len(small_chunks)} chunks below minimum size")

        # Check for very large chunks
        large_chunks = [c for c in chunks if c.metadata.char_count > self.chunk_size * 1.5]
        if large_chunks:
            logger.warning(f"Found {len(large_chunks)} chunks significantly above target size")

        # Log statistics
        avg_tokens = sum(c.metadata.token_count for c in chunks) / len(chunks)
        avg_chars = sum(c.metadata.char_count for c in chunks) / len(chunks)

        logger.info(f"Chunk statistics: avg_tokens={avg_tokens:.0f}, avg_chars={avg_chars:.0f}")

    def save_to_json(self, chunks: List[HierarchicalChunk], output_path: str):
        """Save chunks to JSON file."""
        output_path = Path(output_path)

        data = {
            'total_chunks': len(chunks),
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'chunks': [
                {
                    'content': chunk.content,
                    'metadata': asdict(chunk.metadata)
                }
                for chunk in chunks
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(chunks)} chunks to: {output_path}")

    @staticmethod
    def load_from_json(json_path: str) -> List[HierarchicalChunk]:
        """Load chunks from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = [
            HierarchicalChunk(
                content=item['content'],
                metadata=ChunkMetadata(**item['metadata'])
            )
            for item in data['chunks']
        ]

        return chunks


def main():
    """Example usage of HierarchicalChunker."""
    from src.ingestion.docling_processor import DoclingPDFProcessor
    from config.settings import get_settings

    try:
        settings = get_settings()
    except:
        print("⚠️  Could not load settings. Using default paths.")
        processed_path = "cache/docling_processed.json"
        chunks_path = "cache/hierarchical_chunks.json"
    else:
        processed_path = settings.processed_pdf_path
        chunks_path = settings.chunks_path

    # Load processed document
    print(f"Loading processed document from: {processed_path}")
    structure = DoclingPDFProcessor.load_from_json(str(processed_path))

    # Chunk document
    chunker = HierarchicalChunker(
        chunk_size=1500,
        chunk_overlap=300,
        min_chunk_size=200
    )

    chunks = chunker.chunk_document(
        structure,
        include_images=True,
        include_tables=True
    )

    # Save chunks
    chunker.save_to_json(chunks, str(chunks_path))

    # Print summary
    print("\n" + "="*60)
    print("📦 CHUNKING SUMMARY")
    print("="*60)
    print(f"Total Chunks: {len(chunks)}")
    print(f"Chunk Size: {chunker.chunk_size} chars")
    print(f"Chunk Overlap: {chunker.chunk_overlap} chars")

    # Statistics
    with_images = sum(1 for c in chunks if c.metadata.has_images)
    with_tables = sum(1 for c in chunks if c.metadata.has_tables)
    with_code = sum(1 for c in chunks if c.metadata.has_code)
    continuations = sum(1 for c in chunks if c.metadata.is_continuation)

    print(f"\nContent Statistics:")
    print(f"  Chunks with images: {with_images}")
    print(f"  Chunks with tables: {with_tables}")
    print(f"  Chunks with code: {with_code}")
    print(f"  Continuation chunks: {continuations}")

    # Technical depth
    depth_counts = defaultdict(int)
    for c in chunks:
        depth_counts[c.metadata.technical_depth] += 1

    print(f"\nTechnical Depth:")
    for depth, count in sorted(depth_counts.items()):
        print(f"  {depth}: {count}")

    print("="*60 + "\n")

    # Show sample chunk
    print("📄 Sample Chunk:")
    sample = chunks[10] if len(chunks) > 10 else chunks[0]
    print(f"\nChunk ID: {sample.metadata.chunk_id}")
    print(f"Heading Path: {' > '.join(sample.metadata.heading_path)}")
    print(f"Pages: {sample.metadata.page_start}-{sample.metadata.page_end}")
    print(f"Tokens: {sample.metadata.token_count}")
    print(f"Content:\n{sample.content[:500]}...")

    print("\n✅ Chunking complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
