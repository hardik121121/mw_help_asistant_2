"""
Advanced PDF Processing with Docling.
Extracts hierarchical structure, tables, images, and metadata from PDF documents.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import time

try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("⚠️  Docling not installed. Please run: pip install docling")
    print("Falling back to basic processing mode...")
    DocumentConverter = None

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


class DoclingPDFProcessor:
    """
    Advanced PDF processor using Docling for structure-aware extraction.
    Preserves document hierarchy, extracts tables, images, and metadata.
    """

    def __init__(
        self,
        pdf_path: str,
        image_output_dir: str = "cache/images",
        enable_tables: bool = True,
        enable_images: bool = True,
        enable_ocr: bool = False,  # Set to True if PDF has scanned images
    ):
        """
        Initialize Docling PDF processor.

        Args:
            pdf_path: Path to the PDF file
            image_output_dir: Directory to save extracted images
            enable_tables: Whether to extract tables
            enable_images: Whether to extract images
            enable_ocr: Whether to use OCR for scanned documents
        """
        self.pdf_path = Path(pdf_path)
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_tables = enable_tables
        self.enable_images = enable_images
        self.enable_ocr = enable_ocr

        if DocumentConverter is None:
            raise ImportError(
                "Docling is not installed. Please run: pip install docling"
            )

        logger.info(f"Initialized DoclingPDFProcessor for: {self.pdf_path}")

    def process(self) -> DocumentStructure:
        """
        Process the PDF and extract all content with structure.

        Returns:
            DocumentStructure object with all extracted content
        """
        logger.info(f"Starting PDF processing: {self.pdf_path}")
        start_time = time.time()

        # Initialize converter with default options
        converter = DocumentConverter()

        # Convert document
        logger.info("Converting PDF with Docling...")
        result = converter.convert(str(self.pdf_path))
        doc = result.document

        logger.info(f"PDF converted successfully. Pages: {len(doc.pages)}")

        # Extract structure
        structure = DocumentStructure(
            total_pages=len(doc.pages),
            title=self._extract_title(doc),
            metadata=self._extract_metadata(doc)
        )

        # Process each page
        heading_stack = []  # Track heading hierarchy
        current_headings = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}

        for page_idx, page in enumerate(doc.pages, start=1):
            logger.info(f"Processing page {page_idx}/{len(doc.pages)}")

            # Safety check: skip if page is not an object
            if not hasattr(page, 'elements'):
                logger.warning(f"Skipping page {page_idx}: not a valid page object")
                continue

            # Extract text blocks with hierarchy
            for element in page.elements:
                text_block = self._process_element(
                    element,
                    page_idx,
                    current_headings
                )

                if text_block:
                    structure.text_blocks.append(text_block)

                    # Update heading hierarchy
                    if text_block.block_type == 'heading' and text_block.level:
                        current_headings[text_block.level] = text_block.text
                        # Clear lower-level headings
                        for level in range(text_block.level + 1, 7):
                            current_headings[level] = None

            # Extract images
            if self.enable_images and hasattr(page, 'images'):
                for img_idx, image in enumerate(page.images):
                    image_data = self._extract_image(image, page_idx, img_idx)
                    if image_data:
                        structure.images.append(image_data)

            # Extract tables
            if self.enable_tables and hasattr(page, 'tables'):
                for table_idx, table in enumerate(page.tables):
                    table_data = self._extract_table(table, page_idx, table_idx)
                    if table_data:
                        structure.tables.append(table_data)

        # Build table of contents
        structure.toc = self._build_toc(structure.text_blocks)

        elapsed_time = time.time() - start_time
        logger.info(f"PDF processing complete in {elapsed_time:.2f}s")
        logger.info(f"Extracted: {len(structure.text_blocks)} text blocks, "
                   f"{len(structure.images)} images, {len(structure.tables)} tables")

        return structure

    def _extract_title(self, doc) -> Optional[str]:
        """Extract document title from metadata or first heading."""
        # Try metadata first
        if hasattr(doc, 'metadata') and 'title' in doc.metadata:
            return doc.metadata['title']

        # Fallback to first H1 heading
        try:
            if hasattr(doc, 'pages') and doc.pages:
                for page in doc.pages:
                    # Skip if page is not an object (safety check)
                    if not hasattr(page, 'elements'):
                        continue
                    for element in page.elements:
                        if hasattr(element, 'label') and element.label == 'title':
                            return element.text
                        if hasattr(element, 'level') and element.level == 1:
                            return element.text
        except Exception as e:
            logger.warning(f"Failed to extract title from pages: {e}")

        return None

    def _extract_metadata(self, doc) -> Dict:
        """Extract document metadata."""
        metadata = {}

        if hasattr(doc, 'metadata'):
            metadata.update(doc.metadata)

        return metadata

    def _process_element(
        self,
        element,
        page_number: int,
        current_headings: Dict[int, Optional[str]]
    ) -> Optional[TextBlock]:
        """
        Process a single document element and extract text with hierarchy.

        Args:
            element: Docling element
            page_number: Current page number
            current_headings: Dictionary tracking current heading at each level

        Returns:
            TextBlock object or None
        """
        if not hasattr(element, 'text') or not element.text:
            return None

        # Determine block type and level
        block_type = 'paragraph'
        level = None

        if hasattr(element, 'label'):
            label = element.label.lower()

            if 'heading' in label or 'title' in label:
                block_type = 'heading'
                # Extract level from label (e.g., 'heading_1', 'heading-2')
                for i in range(1, 7):
                    if str(i) in label or f'h{i}' in label:
                        level = i
                        break
                if level is None:
                    level = 1  # Default to H1 if not specified

            elif 'list' in label:
                block_type = 'list_item'
            elif 'table' in label:
                block_type = 'table'
            elif 'caption' in label or 'figure' in label:
                block_type = 'caption'
            elif 'code' in label:
                block_type = 'code'

        # Build heading path (hierarchical context)
        heading_path = []
        if block_type == 'heading':
            # Include all parent headings
            for lvl in range(1, level):
                if current_headings.get(lvl):
                    heading_path.append(current_headings[lvl])
        else:
            # For non-headings, include all current headings
            for lvl in range(1, 7):
                if current_headings.get(lvl):
                    heading_path.append(current_headings[lvl])

        # Get parent heading (immediate parent)
        parent_heading = None
        if block_type != 'heading':
            # Find the most recent heading
            for lvl in range(6, 0, -1):
                if current_headings.get(lvl):
                    parent_heading = current_headings[lvl]
                    break

        # Extract bounding box if available
        bbox = None
        if hasattr(element, 'bbox'):
            bbox = (element.bbox.l, element.bbox.t, element.bbox.r, element.bbox.b)

        # Additional metadata
        metadata = {}
        if hasattr(element, 'font_size'):
            metadata['font_size'] = element.font_size
        if hasattr(element, 'font_name'):
            metadata['font_name'] = element.font_name
        if hasattr(element, 'confidence'):
            metadata['confidence'] = element.confidence

        return TextBlock(
            text=element.text.strip(),
            block_type=block_type,
            level=level,
            page_number=page_number,
            bbox=bbox,
            parent_heading=parent_heading,
            heading_path=heading_path,
            metadata=metadata
        )

    def _extract_image(
        self,
        image_element,
        page_number: int,
        image_index: int
    ) -> Optional[ImageData]:
        """Extract image and save to disk."""
        try:
            # Get image data
            if hasattr(image_element, 'image'):
                img = image_element.image
            elif hasattr(image_element, 'pil_image'):
                img = image_element.pil_image
            else:
                return None

            # Generate filename
            filename = f"page_{page_number:04d}_img_{image_index:02d}.png"
            file_path = self.image_output_dir / filename

            # Save image
            if isinstance(img, Image.Image):
                img.save(file_path)
            else:
                # Convert bytes to PIL Image
                img_pil = Image.open(io.BytesIO(img))
                img_pil.save(file_path)

            # Extract caption if available
            caption = None
            if hasattr(image_element, 'caption'):
                caption = image_element.caption

            # Get dimensions
            width, height = img.size if isinstance(img, Image.Image) else (0, 0)

            # Get bounding box
            bbox = None
            if hasattr(image_element, 'bbox'):
                bbox = (
                    image_element.bbox.l,
                    image_element.bbox.t,
                    image_element.bbox.r,
                    image_element.bbox.b
                )

            return ImageData(
                page_number=page_number,
                image_index=image_index,
                file_path=str(file_path),
                caption=caption,
                width=width,
                height=height,
                bbox=bbox
            )

        except Exception as e:
            logger.warning(f"Failed to extract image on page {page_number}: {e}")
            return None

    def _extract_table(
        self,
        table_element,
        page_number: int,
        table_index: int
    ) -> Optional[TableData]:
        """Extract table structure and content."""
        try:
            # Get table HTML if available
            html = None
            if hasattr(table_element, 'export_to_html'):
                html = table_element.export_to_html()
            elif hasattr(table_element, 'html'):
                html = table_element.html

            # Get table text
            text = None
            if hasattr(table_element, 'export_to_markdown'):
                text = table_element.export_to_markdown()
            elif hasattr(table_element, 'text'):
                text = table_element.text

            # Get dimensions
            rows = 0
            cols = 0
            if hasattr(table_element, 'num_rows'):
                rows = table_element.num_rows
            if hasattr(table_element, 'num_cols'):
                cols = table_element.num_cols

            # Get caption
            caption = None
            if hasattr(table_element, 'caption'):
                caption = table_element.caption

            return TableData(
                page_number=page_number,
                table_index=table_index,
                html=html,
                text=text,
                rows=rows,
                cols=cols,
                caption=caption
            )

        except Exception as e:
            logger.warning(f"Failed to extract table on page {page_number}: {e}")
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
    """Example usage of DoclingPDFProcessor."""
    import sys
    from config.settings import get_settings

    try:
        settings = get_settings()
    except:
        print("⚠️  Could not load settings. Using default paths.")
        pdf_path = "data/helpdocs.pdf"
        output_path = "cache/docling_processed.json"
        image_dir = "cache/images"
    else:
        pdf_path = settings.pdf_path
        output_path = settings.processed_pdf_path
        image_dir = settings.image_cache_dir

    # Process PDF
    processor = DoclingPDFProcessor(
        pdf_path=str(pdf_path),
        image_output_dir=str(image_dir),
        enable_tables=True,
        enable_images=False,  # Disabled for faster processing
        enable_ocr=False
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

    print("\n✅ Processing complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()
