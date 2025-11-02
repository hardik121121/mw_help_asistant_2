"""
Advanced PDF Processing with PyMuPDF.
Extracts hierarchical structure, tables, images, and metadata from PDF documents.
Faster alternative to Docling with custom structure detection.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import time

try:
    import fitz  # PyMuPDF
except ImportError:
    print("⚠️  PyMuPDF not installed. Please run: pip install pymupdf")
    fitz = None

from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Represents an extracted image with metadata."""
    page_number: int
    image_index: int  # Index on the page
    file_path: str
    caption: Optional[str] = None
    width: int = 0
    height: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)


@dataclass
class TableData:
    """Represents an extracted table with structure."""
    page_number: int
    table_index: int
    html: Optional[str] = None
    text: Optional[str] = None
    rows: int = 0
    cols: int = 0
    caption: Optional[str] = None


@dataclass
class TextBlock:
    """Represents a text block with hierarchical information."""
    text: str
    block_type: str  # 'heading', 'paragraph', 'list_item', 'table', etc.
    level: Optional[int] = None  # Heading level (1-6)
    page_number: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    parent_heading: Optional[str] = None
    heading_path: List[str] = field(default_factory=list)  # Full path from root
    metadata: Dict = field(default_factory=dict)


@dataclass
class DocumentStructure:
    """Complete document structure with all extracted content."""
    title: Optional[str] = None
    total_pages: int = 0
    text_blocks: List[TextBlock] = field(default_factory=list)
    images: List[ImageData] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    toc: List[Dict] = field(default_factory=list)  # Table of contents
    metadata: Dict = field(default_factory=dict)


class PyMuPDFProcessor:
    """
    Advanced PDF processor using PyMuPDF for structure-aware extraction.
    Preserves document hierarchy through font size analysis.
    """

    def __init__(
        self,
        pdf_path: str,
        image_output_dir: str = "cache/images",
        enable_tables: bool = True,
        enable_images: bool = True,
        heading_size_thresholds: Optional[Dict[int, float]] = None,
    ):
        """
        Initialize PyMuPDF PDF processor.

        Args:
            pdf_path: Path to the PDF file
            image_output_dir: Directory to save extracted images
            enable_tables: Whether to extract tables
            enable_images: Whether to extract images
            heading_size_thresholds: Dict mapping heading levels to min font sizes
                                    e.g., {1: 20, 2: 16, 3: 14, 4: 12}
        """
        self.pdf_path = Path(pdf_path)
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_tables = enable_tables
        self.enable_images = enable_images

        # Default heading thresholds from .env or defaults
        if heading_size_thresholds is None:
            from config.settings import get_settings
            settings = get_settings()
            self.heading_thresholds = {
                1: settings.heading_1_size,
                2: settings.heading_2_size,
                3: settings.heading_3_size,
                4: settings.heading_4_size,
            }
        else:
            self.heading_thresholds = heading_size_thresholds

        if fitz is None:
            raise ImportError(
                "PyMuPDF is not installed. Please run: pip install pymupdf"
            )

        logger.info(f"Initialized PyMuPDFProcessor for: {self.pdf_path}")
        logger.info(f"Heading thresholds: {self.heading_thresholds}")

    def process(self) -> DocumentStructure:
        """
        Process the PDF and extract all content with structure.

        Returns:
            DocumentStructure object with all extracted content
        """
        logger.info(f"Starting PDF processing: {self.pdf_path}")
        start_time = time.time()

        # Open PDF
        doc = fitz.open(str(self.pdf_path))
        logger.info(f"PDF opened successfully. Pages: {len(doc)}")

        # Extract structure
        structure = DocumentStructure(
            total_pages=len(doc),
            title=self._extract_title(doc),
            metadata=self._extract_metadata(doc)
        )

        # Track heading hierarchy
        current_headings = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            logger.info(f"Processing page {page_num + 1}/{len(doc)}")

            # Extract text blocks with hierarchy
            text_blocks = self._extract_text_blocks(page, page_num + 1)

            for text_block in text_blocks:
                # Build heading path
                text_block.heading_path = self._build_heading_path(
                    current_headings,
                    text_block.level if text_block.block_type == 'heading' else None
                )

                # Set parent heading
                if text_block.block_type != 'heading':
                    text_block.parent_heading = self._get_parent_heading(current_headings)

                structure.text_blocks.append(text_block)

                # Update heading hierarchy
                if text_block.block_type == 'heading' and text_block.level:
                    current_headings[text_block.level] = text_block.text
                    # Clear lower-level headings
                    for level in range(text_block.level + 1, 7):
                        current_headings[level] = None

            # Extract images
            if self.enable_images:
                images = self._extract_images(page, page_num + 1)
                structure.images.extend(images)

            # Extract tables (basic detection)
            if self.enable_tables:
                tables = self._extract_tables(page, page_num + 1)
                structure.tables.extend(tables)

        # Build table of contents
        structure.toc = self._build_toc(structure.text_blocks)

        doc.close()

        elapsed_time = time.time() - start_time
        logger.info(f"PDF processing complete in {elapsed_time:.2f}s")
        logger.info(f"Extracted: {len(structure.text_blocks)} text blocks, "
                   f"{len(structure.images)} images, {len(structure.tables)} tables")

        return structure

    def _extract_title(self, doc) -> Optional[str]:
        """Extract document title from metadata or first large text."""
        # Try metadata first
        if doc.metadata and doc.metadata.get('title'):
            return doc.metadata['title']

        # Fallback to first large text (likely title)
        try:
            first_page = doc[0]
            blocks = first_page.get_text("dict")["blocks"]

            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            size = span["size"]

                            # Title is usually the largest text on first page
                            if size >= self.heading_thresholds[1] and len(text) > 10:
                                return text
        except Exception as e:
            logger.warning(f"Failed to extract title: {e}")

        return None

    def _extract_metadata(self, doc) -> Dict:
        """Extract document metadata."""
        metadata = {}

        if doc.metadata:
            metadata.update({
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'mod_date': doc.metadata.get('modDate', ''),
            })

        return metadata

    def _determine_block_type(self, text: str, font_size: float, font_flags: int) -> Tuple[str, Optional[int]]:
        """
        Determine block type and heading level based on font characteristics.

        Args:
            text: Text content
            font_size: Font size in points
            font_flags: PyMuPDF font flags (bit field)

        Returns:
            Tuple of (block_type, heading_level)
        """
        # Check if bold (bit 4 set in font_flags)
        is_bold = bool(font_flags & (1 << 4))

        # Detect heading based on font size
        if font_size >= self.heading_thresholds[1]:
            return 'heading', 1
        elif font_size >= self.heading_thresholds[2]:
            return 'heading', 2
        elif font_size >= self.heading_thresholds[3]:
            return 'heading', 3
        elif font_size >= self.heading_thresholds[4] and is_bold:
            return 'heading', 4

        # Detect list items
        if text.strip() and (
            text.strip()[0] in ['•', '○', '■', '▪', '–', '-', '*'] or
            (len(text.strip()) > 2 and text.strip()[1:3] in ['. ', ') '])
        ):
            return 'list_item', None

        # Default to paragraph
        return 'paragraph', None

    def _extract_text_blocks(self, page, page_number: int) -> List[TextBlock]:
        """Extract text blocks from a page with structure detection."""
        text_blocks = []

        # Get text with detailed formatting info
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            # Get block bbox
            bbox = (block["bbox"][0], block["bbox"][1], block["bbox"][2], block["bbox"][3])

            # Combine all text in block
            block_text = ""
            avg_font_size = 0
            font_sizes = []
            font_flags_list = []
            font_names = []

            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                    font_sizes.append(span["size"])
                    font_flags_list.append(span["flags"])
                    font_names.append(span["font"])

                block_text += line_text.strip() + " "

            block_text = block_text.strip()

            if not block_text:
                continue

            # Calculate average font size
            if font_sizes:
                avg_font_size = sum(font_sizes) / len(font_sizes)

            # Determine block type
            most_common_flags = max(set(font_flags_list), key=font_flags_list.count) if font_flags_list else 0
            block_type, heading_level = self._determine_block_type(
                block_text,
                avg_font_size,
                most_common_flags
            )

            # Create TextBlock
            text_block = TextBlock(
                text=block_text,
                block_type=block_type,
                level=heading_level,
                page_number=page_number,
                bbox=bbox,
                metadata={
                    'font_size': avg_font_size,
                    'font_name': font_names[0] if font_names else None,
                    'is_bold': bool(most_common_flags & (1 << 4)),
                    'is_italic': bool(most_common_flags & (1 << 1)),
                }
            )

            text_blocks.append(text_block)

        return text_blocks

    def _extract_images(self, page, page_number: int) -> List[ImageData]:
        """Extract images from a page."""
        images = []

        try:
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]

                # Get image
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]

                # Save image
                filename = f"page_{page_number:04d}_img_{img_idx:02d}.png"
                file_path = self.image_output_dir / filename

                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                # Get image dimensions
                img_pil = Image.open(io.BytesIO(image_bytes))
                width, height = img_pil.size

                # Get image bbox on page
                img_rects = page.get_image_rects(xref)
                bbox = None
                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

                images.append(ImageData(
                    page_number=page_number,
                    image_index=img_idx,
                    file_path=str(file_path),
                    width=width,
                    height=height,
                    bbox=bbox
                ))

        except Exception as e:
            logger.warning(f"Failed to extract images from page {page_number}: {e}")

        return images

    def _extract_tables(self, page, page_number: int) -> List[TableData]:
        """
        Basic table detection based on text alignment.
        Note: This is a simplified approach. For better table extraction,
        consider using camelot-py or tabula-py.
        """
        tables = []

        # For now, we'll skip advanced table detection
        # This can be enhanced later with proper table detection libraries

        return tables

    def _build_heading_path(self, current_headings: Dict[int, Optional[str]], current_level: Optional[int]) -> List[str]:
        """Build hierarchical path of headings."""
        path = []

        if current_level is None:
            # For non-headings, include all current headings
            for level in range(1, 7):
                if current_headings.get(level):
                    path.append(current_headings[level])
        else:
            # For headings, include only parent headings
            for level in range(1, current_level):
                if current_headings.get(level):
                    path.append(current_headings[level])

        return path

    def _get_parent_heading(self, current_headings: Dict[int, Optional[str]]) -> Optional[str]:
        """Get the most recent heading (immediate parent)."""
        for level in range(6, 0, -1):
            if current_headings.get(level):
                return current_headings[level]
        return None

    def _build_toc(self, text_blocks: List[TextBlock]) -> List[Dict]:
        """Build table of contents from heading blocks."""
        toc = []

        for block in text_blocks:
            if block.block_type == 'heading' and block.level:
                toc.append({
                    'text': block.text,
                    'level': block.level,
                    'page': block.page_number,
                    'heading_path': block.heading_path
                })

        return toc

    def save_to_json(self, structure: DocumentStructure, output_path: str):
        """Save extracted structure to JSON file."""
        output_path = Path(output_path)

        # Convert dataclasses to dictionaries
        data = {
            'title': structure.title,
            'total_pages': structure.total_pages,
            'text_blocks': [asdict(block) for block in structure.text_blocks],
            'images': [asdict(img) for img in structure.images],
            'tables': [asdict(table) for table in structure.tables],
            'toc': structure.toc,
            'metadata': structure.metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved document structure to: {output_path}")

    @staticmethod
    def load_from_json(json_path: str) -> DocumentStructure:
        """Load document structure from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        structure = DocumentStructure(
            title=data.get('title'),
            total_pages=data.get('total_pages', 0),
            text_blocks=[TextBlock(**block) for block in data.get('text_blocks', [])],
            images=[ImageData(**img) for img in data.get('images', [])],
            tables=[TableData(**table) for table in data.get('tables', [])],
            toc=data.get('toc', []),
            metadata=data.get('metadata', {})
        )

        return structure


def main():
    """Example usage of PyMuPDFProcessor."""
    import sys
    from config.settings import get_settings

    try:
        settings = get_settings()
    except:
        print("⚠️  Could not load settings. Using default paths.")
        pdf_path = "data/helpdocs.pdf"
        output_path = "cache/pymupdf_processed.json"
        image_dir = "cache/images"
    else:
        pdf_path = settings.pdf_path
        output_path = settings.processed_pdf_path
        image_dir = settings.image_cache_dir

    # Process PDF
    processor = PyMuPDFProcessor(
        pdf_path=str(pdf_path),
        image_output_dir=str(image_dir),
        enable_tables=True,
        enable_images=True,  # Enabled for full processing
    )

    structure = processor.process()

    # Save to JSON
    processor.save_to_json(structure, str(output_path))

    # Print summary
    print("\n" + "="*60)
    print("📄 PDF PROCESSING SUMMARY")
    print("="*60)
    print(f"Title: {structure.title}")
    print(f"Total Pages: {structure.total_pages}")
    print(f"Text Blocks: {len(structure.text_blocks)}")
    print(f"Images: {len(structure.images)}")
    print(f"Tables: {len(structure.tables)}")
    print(f"TOC Entries: {len(structure.toc)}")
    print("="*60 + "\n")

    # Print TOC preview
    print("📑 Table of Contents (first 20 entries):")
    for i, entry in enumerate(structure.toc[:20], 1):
        indent = "  " * (entry['level'] - 1)
        print(f"{i}. {indent}{entry['text']} (Page {entry['page']})")

    # Print sample text blocks
    print("\n📝 Sample Text Blocks (first 10):")
    for i, block in enumerate(structure.text_blocks[:10], 1):
        type_label = f"{block.block_type}"
        if block.level:
            type_label += f" (H{block.level})"
        print(f"\n{i}. [{type_label}] Page {block.page_number}")
        print(f"   Text: {block.text[:100]}..." if len(block.text) > 100 else f"   Text: {block.text}")
        if block.heading_path:
            print(f"   Path: {' > '.join(block.heading_path)}")

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
