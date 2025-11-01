# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maximum-quality RAG system for complex multi-topic queries across 2300+ pages of Watermelon documentation. Implements **query decomposition + hierarchical chunking + multi-step retrieval** for questions spanning multiple topics.

**Example**: *"How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"* requires understanding 3-4 different topics, retrieving context from different sections, and integrating information into coherent step-by-step answers.

**Current Status**: Phase 2 complete (22% overall - document processing pipeline ready). Phases 3-9 (query understanding, retrieval, generation, UI) are planned but not implemented.

## Quick Reference

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Then add your API keys

# Run Phase 2 pipeline (document processing)
python -m src.ingestion.docling_processor      # 15-60 min
python -m src.ingestion.hierarchical_chunker   # 1-2 min
python -m src.ingestion.chunk_evaluator        # <1 min

# Check progress
ls -lh cache/docling_processed.json cache/hierarchical_chunks.json
cat tests/results/chunk_quality_report.txt

# Validate configuration
python config/settings.py
```

**Most Common Error**: Running modules with `python file.py` instead of `python -m module.path` → causes import errors.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Validate configuration
python config/settings.py
```

### Running Phase 2 Components (Document Processing)
```bash
# CRITICAL: Always use 'python -m module.path' syntax, never direct file paths
# This ensures proper package context for relative imports

python -m src.ingestion.docling_processor      # Extract PDF structure (~15-60 min)
python -m src.ingestion.hierarchical_chunker   # Create context-aware chunks (~1-2 min)
python -m src.ingestion.chunk_evaluator        # Evaluate chunk quality (<1 min)

# Or run full pipeline sequentially
python -m src.ingestion.docling_processor && \
python -m src.ingestion.hierarchical_chunker && \
python -m src.ingestion.chunk_evaluator
```

### Checking Processing State
```bash
# Check what's been processed (each step produces output files)
ls -lh cache/docling_processed.json           # Should be ~23 MB (from step 1)
ls -lh cache/hierarchical_chunks.json         # Should be ~3 MB (from step 2)
ls tests/results/chunk_quality_report.txt     # Quality report (from step 3)

# Check extracted images (if image processing enabled)
ls cache/images/ | wc -l                      # Count extracted images

# View test queries structure
cat tests/test_queries.json | head -50        # See first test query
```

## Architecture & Critical Concepts

### 1. Module Import Pattern (CRITICAL)

**Always use `python -m module.path` syntax**, never direct file paths. This is the #1 most common error when working with this codebase.

```bash
# ✅ CORRECT
python -m src.ingestion.docling_processor
python -m src.ingestion.hierarchical_chunker

# ❌ WRONG - Will fail with ModuleNotFoundError
python src/ingestion/docling_processor.py
python src/ingestion/hierarchical_chunker.py
```

**Root cause**: All modules use relative imports (`from src.ingestion.docling_processor import DocumentStructure`) which require proper package context. The `-m` flag ensures Python treats the directory as a package.

### 2. Pydantic V2 Configuration System

The configuration system in `config/settings.py` uses **Pydantic V2** with specific patterns:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    # Direct field definitions (no Field(..., env="VAR"))
    openai_api_key: str
    chunk_size: int = 1500

    # V2 config pattern
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    # V2 validator pattern
    @field_validator("openai_api_key")
    @classmethod
    def check_not_placeholder(cls, v: str) -> str:
        # validation logic
        return v
```

**Important**: Settings fields are **flat** (not nested). Access as:
- `settings.pdf_path` ✅ NOT `settings.paths.pdf_path` ❌
- `settings.chunk_size` ✅ NOT `settings.document.chunk_size` ❌

### 3. Hierarchical Document Processing Pipeline

The system processes documents in 3 stages:

```
PDF (157 MB, 2257 pages)
    ↓
[Docling Processor] - Extracts structure, headings, tables, images
    ↓
Structured JSON (23 MB) + Images
    ↓
[Hierarchical Chunker] - Creates context-aware chunks with metadata
    ↓
Chunks with Context (3 MB, ~2500 chunks)
    ↓
[Quality Evaluator] - Validates chunk quality
    ↓
Quality Report + Metrics
```

**Key Innovation**: Unlike traditional RAG that loses document structure, this system:
- Maintains heading hierarchy (H1→H2→H3→H4) in every chunk
- Prepends section path for context: `"Section: Getting Started > Integration > MS Teams\n\n[content]"`
- Preserves semantic boundaries (doesn't split mid-section)
- Adds 20+ metadata fields per chunk

### 4. Docling Processing Performance

**Important**: Docling PDF processing is CPU-intensive and can take 15-60 minutes for large PDFs (2000+ pages).

**Why it's slow**:
- Extracts hierarchical structure (not just raw text)
- Processes images and tables
- Runs OCR on embedded images
- Preserves bounding boxes for all elements

**Optimization for testing**:
```python
processor = DoclingPDFProcessor(
    pdf_path="data/helpdocs.pdf",
    enable_tables=True,      # Keep tables
    enable_images=False,     # Disable to speed up (if not needed immediately)
    enable_ocr=False         # Disable if PDF has selectable text
)
```

**Default Docling Configuration**:
The code uses `DocumentConverter()` with defaults (StandardPdfPipeline), which includes:
- OCR with RapidOCR
- Table extraction
- Image extraction
- Structure preservation

### 5. Chunk Metadata Schema (20+ Fields)

Each `HierarchicalChunk` has a `ChunkMetadata` dataclass with rich metadata (see `src/ingestion/hierarchical_chunker.py:28`):

```python
@dataclass
class ChunkMetadata:
    # Location
    chunk_id: str                    # Unique identifier
    page_start: int                  # Starting page number
    page_end: int                    # Ending page number
    section_id: str                  # Section identifier

    # Hierarchy (enables context reconstruction)
    heading_path: List[str]          # ["Chapter 1", "Section 1.1", "Subsection 1.1.1"]
    current_heading: Optional[str]   # "Subsection 1.1.1"
    heading_level: Optional[int]     # 1-4 (H1-H4)

    # Content characteristics (enables smart filtering)
    content_type: str                # "text" | "table" | "list" | "code" | "mixed"
    technical_depth: str             # "low" | "medium" | "high"

    # Feature flags (enables multi-modal retrieval)
    has_images: bool
    has_tables: bool
    has_code: bool
    has_lists: bool

    # References to related content
    image_paths: List[str]           # Paths to extracted images
    image_captions: List[str]        # Image captions
    table_texts: List[str]           # Table content summaries

    # Chunking metadata
    is_continuation: bool            # True if part of split section
    chunk_index: int                 # Index within section
    total_chunks_in_section: int     # Total chunks in this section

    # Size information
    token_count: int                 # OpenAI tokens (for cost estimation)
    char_count: int                  # Character count
```

**Usage in retrieval**: This metadata enables sophisticated filtering (e.g., "only return chunks with code examples" or "prioritize high technical depth") and ranking (e.g., "boost chunks with images for visual queries").

## Data Flow & File Structure

### Input/Output Files

**Input**:
- `data/helpdocs.pdf` - Source documentation (157 MB, 2257 pages)
- `.env` - API keys and configuration (not in git)

**Output from Phase 2**:
- `cache/docling_processed.json` - Structured document (23 MB)
- `cache/hierarchical_chunks.json` - Chunks with metadata (3 MB)
- `cache/images/*.png` - Extracted images
- `tests/results/chunk_quality_report.txt` - Quality evaluation

**Configuration**:
- `.env.example` - Template with all config options
- `config/settings.py` - Pydantic settings class

**Test Data**:
- `tests/test_queries.json` - 30 complex multi-topic queries (structure shown below)

### Test Query Structure

Each test query in `tests/test_queries.json` follows this format:

```json
{
  "id": 1,
  "query": "How do I create a no-code block on Watermelon platform and process it for Autonomous Functional Testing?",
  "type": "multi-topic_procedural",
  "complexity": "high",
  "topics": ["no-code blocks", "autonomous functional testing", "workflow creation"],
  "expected_components": [
    "What are no-code blocks",
    "Steps to create a no-code block",
    "What is Autonomous Functional Testing",
    "How to connect blocks to testing framework"
  ]
}
```

Query types include: `multi-topic_procedural`, `multi-topic_integration`, `conceptual_procedural`, `troubleshooting`, `security_compliance`.

### Module Organization

```
src/
├── ingestion/          # ✅ Phase 2 COMPLETE
│   ├── docling_processor.py    # PDF → Structured JSON (Docling-based)
│   ├── hierarchical_chunker.py # JSON → Context-aware chunks
│   └── chunk_evaluator.py      # Quality evaluation
├── query/              # 🚧 Phase 3 NOT STARTED
├── retrieval/          # 🚧 Phase 4 NOT STARTED
├── database/           # 🚧 Phase 5 NOT STARTED
├── generation/         # 🚧 Phase 6 NOT STARTED
└── memory/             # 🚧 Phase 8 NOT STARTED
```

**Progress**: 2/9 phases complete (22%). Only document processing pipeline is implemented.

## Critical Implementation Details

### Context Injection Pattern

The hierarchical chunker **prepends** section hierarchy to every chunk:

```python
# Example chunk content
"""
Section: Getting Started > Integrations > MS Teams

To integrate MS Teams with Watermelon:
1. Navigate to Settings > Integrations
2. Click "Connect MS Teams"
...
"""
```

This ensures chunks contain hierarchical context even when retrieved in isolation.

### Quality Targets

The chunk evaluator measures quality with these targets:
- **Size consistency score**: >0.85
- **Structure preservation score**: >0.90
- **Context completeness score**: >0.85
- **Overall quality score**: >0.80 (critical threshold)

If quality is below 0.80, review chunking parameters in `.env`:
```bash
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
MIN_CHUNK_SIZE=200
```

### API Cost Considerations

**Free Tier Limits**:
- Groq: 14,400 requests/day (free)
- Pinecone: 100,000 vectors (free)
- Cohere: 1,000 calls/month (then $0.002/call)

**One-time costs**:
- OpenAI embeddings (~2500 chunks): $3-5

**Per query**:
- ~$0.002-0.005 (mostly Cohere re-ranking)

## Common Patterns & Conventions

### Error Handling

All modules use try/except with graceful degradation:

```python
try:
    from docling.document_converter import DocumentConverter
except ImportError:
    print("⚠️  Docling not installed. Please run: pip install docling")
    DocumentConverter = None
```

### Logging

Use Python's logging module (already configured):

```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing page {page_num}/{total_pages}")
logger.warning(f"Failed to extract image: {e}")
```

### Dataclasses for Structured Data

The codebase extensively uses dataclasses:

```python
from dataclasses import dataclass, field, asdict

@dataclass
class ChunkMetadata:
    chunk_id: str
    page_start: int
    heading_path: List[str] = field(default_factory=list)
```

Use `asdict()` for JSON serialization.

### Settings Access

Always access settings via the getter:

```python
from config.settings import get_settings

settings = get_settings()
pdf_path = settings.pdf_path  # NOT settings.paths.pdf_path
```

## Implementing Future Phases (3-9)

### Development Approach

Each new phase follows this pattern:
1. Create module in appropriate `src/` subdirectory
2. Follow existing patterns (dataclasses, logging, error handling)
3. Add to pipeline in sequential order
4. Test with `tests/test_queries.json` queries
5. Update phase progress in this file

### Key Architectural Decisions for Future Phases

**Phase 3 (Query Understanding)** - Create `src/query/`:
- Use OpenAI GPT-4 for query decomposition (via Groq Llama 3.3 70B to save cost)
- Store sub-questions as structured data (dataclass with dependencies)
- Query classification determines retrieval strategy
- Integration point: Takes user query string, returns `DecomposedQuery` object

**Phase 4 (Multi-Step Retrieval)** - Create `src/retrieval/`:
- Hybrid search: Vector (Pinecone) + BM25 (in-memory index)
- Use RRF (Reciprocal Rank Fusion) to merge results: `score = 1 / (k + rank)`
- Cohere reranking as final step
- Context chaining: Use results from sub-question N to refine retrieval for N+1
- Integration point: Takes `DecomposedQuery` + `HierarchicalChunk[]`, returns `RetrievalResult[]`

**Phase 5 (Embeddings)** - Create `src/database/`:
- OpenAI text-embedding-3-large (3072-dim) for all chunks
- Pinecone serverless index (AWS us-east-1, cosine similarity)
- Store chunk metadata in Pinecone for filtering
- BM25 index from chunk content (rank-bm25 library)
- Integration point: One-time embedding generation, then query-time retrieval

**Phase 6 (Generation)** - Create `src/generation/`:
- Groq Llama 3.3 70B for answer generation
- Multi-context prompt template (prepend all relevant chunks)
- Response validation: Check all sub-questions addressed
- Smart image selection from chunk metadata
- Integration point: Takes `RetrievalResult[]`, returns formatted answer with citations

**Phase 7 (Evaluation)** - Extend `tests/`:
- Run all 30 test queries
- Measure: Precision@10, MRR, NDCG (retrieval)
- Measure: Completeness, accuracy, formatting (generation)
- Store results in `tests/results/`

**Phase 8 (UI)** - Create `app.py`:
- Streamlit interface
- Query input → show decomposed sub-questions → show retrieved chunks → final answer
- Debug mode: Display all intermediate steps

See `PROGRESS.md` for detailed specifications of each phase.

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Use `python -m src.module.name` instead of running files directly

### Pydantic Errors
**Problem**: `PydanticDeprecatedSince20` warnings
**Solution**: Settings class already uses Pydantic V2 patterns - don't use `Field(..., env="VAR")` syntax

### Settings AttributeError
**Problem**: `'Settings' object has no attribute 'paths'`
**Solution**: Settings are flat - use `settings.pdf_path` not `settings.paths.pdf_path`

### Docling Slow/Stuck
**Problem**: PDF processing takes >60 minutes
**Solution**:
- For 2000+ page PDFs, 15-60 min is normal on CPU
- To speed up: disable images/OCR if not needed immediately
- Process is not stuck if CPU usage is high (check with `top`)

### Memory Issues
**Problem**: Out of memory during Docling processing
**Solution**: Large PDFs (150+ MB) need 4-8 GB RAM; consider processing in page ranges

## API Keys Required

Get from:
1. **OpenAI**: https://platform.openai.com/api-keys
2. **Pinecone**: https://app.pinecone.io/
3. **Cohere**: https://dashboard.cohere.com/api-keys
4. **Groq**: https://console.groq.com/keys

Add to `.env` (copy from `.env.example`).

## Quality Standards

- **Chunk quality score**: Must be >0.80 before proceeding to embeddings
- **Code documentation**: All functions have comprehensive docstrings
- **Type hints**: Used throughout (Python 3.10+)
- **Error handling**: Graceful degradation with user-friendly messages
- **Logging**: Info/warning/error at appropriate levels

---

## Resuming or Restarting Processing

### If Processing Was Interrupted

Check which outputs exist:
```bash
ls -lh cache/docling_processed.json        # Step 1 output
ls -lh cache/hierarchical_chunks.json      # Step 2 output
ls tests/results/chunk_quality_report.txt  # Step 3 output
```

**Resume from where it stopped**:
- If `docling_processed.json` exists: Skip to step 2 (hierarchical_chunker)
- If `hierarchical_chunks.json` exists: Skip to step 3 (chunk_evaluator)
- If all exist: Phase 2 is complete, ready for Phase 3

### Processing the Full PDF (First Time)

The full 2257-page PDF takes 15-60 minutes depending on features enabled:

```bash
# Standard processing (tables only, no images): ~15-30 min
python -m src.ingestion.docling_processor

# With images: ~60-90 min (modify enable_images=True in docling_processor.py:~490)
```

**To run in background**:
```bash
nohup python -m src.ingestion.docling_processor > logs/docling.log 2>&1 &
tail -f logs/docling.log  # Monitor progress
```

### Testing with Small Subset

For rapid iteration/testing, extract first 20 pages:
```bash
pip install pypdf
python -c "
from PyPDF2 import PdfReader, PdfWriter
reader = PdfReader('data/helpdocs.pdf')
writer = PdfWriter()
for i in range(20):
    writer.add_page(reader.pages[i])
with open('data/helpdocs_test.pdf', 'wb') as f:
    writer.write(f)
"
# Then modify PDF_PATH in .env to point to helpdocs_test.pdf
```

Processing time: <2 minutes for 20 pages.

---

**See Also**:
- `README.md` - Project overview and architecture
- `PROGRESS.md` - Detailed phase breakdowns and technical specs
- `NEXT_STEPS.md` - Current processing options and troubleshooting
- `SETUP.md` - Installation and configuration guide
