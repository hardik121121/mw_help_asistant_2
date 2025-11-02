# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maximum-quality RAG system for complex multi-topic queries across 2300+ pages of Watermelon documentation. Implements **query decomposition + hierarchical chunking + multi-step retrieval + advanced generation** for questions spanning multiple topics.

**Example Query**: *"How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"* requires understanding 3-4 different topics, retrieving context from different sections, and integrating information into coherent step-by-step answers.

**Current Status**: Phases 1-7 complete (78% overall). Full RAG pipeline operational with evaluation framework. Phases 8-9 (UI and Deployment) remain.

## Critical Development Rules

### 1. Module Import Pattern (MOST COMMON ERROR)

**Always use `python -m module.path` syntax**, never direct file paths.

```bash
# ✅ CORRECT
python -m src.ingestion.docling_processor
python -m src.retrieval.multi_step_retriever
python -m src.generation.end_to_end_pipeline
python -m src.evaluation.comprehensive_evaluation

# ❌ WRONG - Will fail with ModuleNotFoundError
python src/ingestion/docling_processor.py
python src/retrieval/multi_step_retriever.py
```

**Root cause**: All modules use relative imports which require proper package context. The `-m` flag ensures Python treats the directory as a package.

### 2. Pydantic V2 Configuration System

Settings fields are **flat** (not nested). Access as:
- `settings.pdf_path` ✅ NOT `settings.paths.pdf_path` ❌
- `settings.chunk_size` ✅ NOT `settings.document.chunk_size` ❌

Always use the getter function:
```python
from config.settings import get_settings

settings = get_settings()
pdf_path = settings.pdf_path
```

### 3. Pinecone Package Version

The project uses the new `pinecone` package (NOT `pinecone-client`). If you see import errors:
```bash
pip uninstall pinecone-client -y
pip install -U pinecone
```

### 4. Groq Rate Limits (CRITICAL)

**Free tier limits**: 100,000 tokens/day for Llama 3.3 70B
- Query understanding uses ~1,000 tokens/query (decomposition)
- Answer generation uses ~6,000 tokens/query
- **Total**: ~7,000 tokens per query
- **Daily capacity**: ~14 queries/day on free tier

**If you hit rate limits**:
```
Error: Rate limit reached for model `llama-3.3-70b-versatile`
Wait time: ~10 minutes (or until daily reset at midnight UTC)
```

**Solutions**:
- Spread testing across multiple days (10 queries/day)
- Upgrade to Groq Dev Tier for higher limits
- Use smaller test sets during development

## Common Commands

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Validate configuration
python -m config.settings
```

### Phase 2: Document Processing (✅ Complete)
```bash
# Process PDF with Docling (~15-60 min)
python -m src.ingestion.docling_processor

# Create hierarchical chunks (~1-2 min)
python -m src.ingestion.hierarchical_chunker

# Evaluate chunk quality (<1 min)
python -m src.ingestion.chunk_evaluator

# Filter TOC chunks (optional)
python -m src.utils.toc_filter --strategy mark
```

**Output**:
- `cache/docling_processed.json` (43 MB)
- `cache/hierarchical_chunks_filtered.json` (4.5 MB, 2,106 content chunks)
- `cache/images/` (1,454 images)

### Phase 3: Query Understanding (✅ Complete)
```bash
# Test individual components
python -m src.query.query_decomposer
python -m src.query.query_classifier
python -m src.query.intent_analyzer

# Test full pipeline
python -m src.query.query_understanding

# Test with actual test queries
python -m src.query.test_phase3
```

**Features**: LLM-based query decomposition, rule-based classification, intent analysis

### Phase 4: Multi-Step Retrieval (✅ Complete)
```bash
# Test individual components
python -m src.retrieval.hybrid_search
python -m src.retrieval.reranker
python -m src.retrieval.context_organizer

# Test full retrieval pipeline
python -m src.retrieval.multi_step_retriever

# Run comprehensive tests
python -m src.retrieval.test_phase4
```

**Features**: Hybrid search (vector + BM25), RRF fusion, Cohere reranking, context organization

### Phase 5: Embeddings & Indexing (✅ Complete)
```bash
# Run full pipeline (embeddings + Pinecone + BM25)
python -m src.database.run_phase5

# Or run individual steps:
python -m src.database.embedding_generator  # Generate embeddings
python -m src.database.vector_store        # Create/upload to Pinecone
python -m src.database.bm25_index         # Create BM25 index
```

**Output**:
- `cache/hierarchical_embeddings.pkl` (59 MB)
- `cache/bm25_index.pkl` (64 MB)
- Pinecone index: `watermelon-docs-v2` (2,106 vectors, 3072-dim)

**Cost**: ~$0.08 for embeddings (OpenAI text-embedding-3-large)

### Phase 6: Advanced Generation (✅ Complete)
```bash
# Test answer generation
python -m src.generation.answer_generator

# Test response validation
python -m src.generation.response_validator

# Test full end-to-end pipeline
python -m src.generation.end_to_end_pipeline
```

**Features**: Multi-strategy generation, response validation, citation extraction, image referencing

### Phase 7: Evaluation & Testing (✅ Complete)
```bash
# Run comprehensive evaluation (interactive)
python -m src.evaluation.comprehensive_evaluation
# Options when prompted:
# - Enter a number (e.g., "5") for first N queries
# - Enter "all" for all 30 queries
# - Press Enter for default (5 queries)

# Test individual metrics
python -m src.evaluation.retrieval_metrics
python -m src.evaluation.generation_metrics
```

**Output**: `tests/results/comprehensive_evaluation.json`

**Important**: Be mindful of Groq rate limits when testing many queries!

### Checking Pipeline State
```bash
# Check what's been processed
ls -lh cache/docling_processed.json              # Phase 2 step 1
ls -lh cache/hierarchical_chunks_filtered.json   # Phase 2 step 2
ls -lh cache/hierarchical_embeddings.pkl         # Phase 5 step 1
ls -lh cache/bm25_index.pkl                      # Phase 5 step 2

# Check quality reports
cat tests/results/chunk_quality_report.txt
cat tests/results/phase3_test_results.json
cat tests/results/comprehensive_evaluation.json

# Check evaluation results
python -c "import json; print(json.dumps(json.load(open('tests/results/comprehensive_evaluation.json'))['statistics'], indent=2))"
```

## Architecture & Key Concepts

### Complete Processing Pipeline (Phases 1-7 Complete)

```
PDF (157 MB, 2257 pages)
    ↓
[Phase 2: Docling Processor]
    → Structured JSON (43 MB) + 1,454 images
    ↓
[Phase 2: Hierarchical Chunker]
    → 2,106 context-aware chunks (4.5 MB)
    ↓
[Phase 5: Embedding Generator]
    → OpenAI embeddings (59 MB, 3072-dim)
    ↓
[Phase 5: Vector Store + BM25]
    → Pinecone index (2,106 vectors)
    → BM25 index (16,460 vocab terms)
    ↓
[Phase 3: Query Understanding]
    ↓
    Query → Decomposed Sub-questions → Classification → Intent
    ↓
[Phase 4: Multi-Step Retrieval]
    ↓
    For each sub-question:
      1. Generate embedding (OpenAI)
      2. Hybrid search (Vector + BM25)
      3. RRF fusion
      4. Cohere reranking
    → Organized context (20 chunks)
    ↓
[Phase 6: Advanced Generation]
    ↓
    1. Strategy-aware prompting (Groq Llama 3.3 70B)
    2. Multi-context integration
    3. Citation extraction
    4. Response validation
    → Final answer with citations & images
    ↓
[Phase 7: Evaluation]
    ↓
    1. Retrieval metrics (Precision, Recall, MRR, MAP, NDCG)
    2. Generation metrics (Completeness, Coherence, Formatting)
    → Performance statistics
```

### Key Innovations

#### 1. **Hierarchical Context Preservation**
Unlike traditional RAG, every chunk includes full section hierarchy:
```python
"""
Section: Getting Started > Integrations > MS Teams

To integrate MS Teams with Watermelon:
1. Navigate to Settings > Integrations
...
"""
```

#### 2. **Rich Chunk Metadata (20+ Fields)**
Each chunk in `hierarchical_chunks_filtered.json` has:
- **Location**: chunk_id, page_start, page_end, section_id
- **Hierarchy**: heading_path, current_heading, heading_level
- **Content flags**: has_images, has_tables, has_code, is_toc
- **Characteristics**: content_type, technical_depth
- **References**: image_paths, table_texts
- **Size**: token_count, char_count

This metadata is preserved in Pinecone for filtering during retrieval.

#### 3. **TOC Filtering**
Pages 1-18 are table of contents. All TOC chunks are marked with `is_toc: true` flag for filtering during retrieval.

#### 4. **Multi-Step Retrieval with Context Chaining**
Each sub-question retrieves independently, then results are:
1. Deduplicated across sub-questions
2. Score-aggregated for multi-retrieved chunks
3. Organized by topic and section hierarchy
4. Context from earlier sub-questions enriches later ones

#### 5. **Strategy-Aware Generation**
Generation adapts based on query type:
- **Step-by-step**: Procedural queries → numbered instructions
- **Comparison**: Feature comparison queries → structured tables
- **Troubleshooting**: Error queries → diagnosis + solutions
- **Standard**: General queries → comprehensive answers

### Data Flow & Files

**Inputs**:
- `data/helpdocs.pdf` (157 MB, 2257 pages)
- `.env` (API keys - not in git)
- `tests/test_queries.json` (30 complex test queries)

**Phase 2 Outputs**:
- `cache/docling_processed.json` - Structured document
- `cache/hierarchical_chunks_filtered.json` - 2,106 chunks with TOC marked
- `cache/images/*.png` - 1,454 extracted images

**Phase 3 Outputs**:
- `tests/results/phase3_test_results.json` - Query understanding tests

**Phase 5 Outputs**:
- `cache/hierarchical_embeddings.pkl` - All embeddings
- `cache/bm25_index.pkl` - Keyword search index
- Pinecone cloud index: `watermelon-docs-v2`

**Phase 7 Outputs**:
- `tests/results/comprehensive_evaluation.json` - Full evaluation results

**Module Organization**:
```
src/
├── ingestion/          # ✅ Phase 2 COMPLETE
│   ├── docling_processor.py       # PDF → structured JSON
│   ├── hierarchical_chunker.py    # JSON → context-aware chunks
│   ├── chunk_evaluator.py         # Quality assessment
│   └── pymupdf_processor.py       # Alternative PDF processor
├── query/              # ✅ Phase 3 COMPLETE
│   ├── query_decomposer.py        # LLM-based decomposition (Groq)
│   ├── query_classifier.py        # Rule-based classification
│   ├── intent_analyzer.py         # Intent extraction
│   ├── query_understanding.py     # Orchestrator
│   └── test_phase3.py            # Test suite
├── database/           # ✅ Phase 5 COMPLETE
│   ├── embedding_generator.py     # OpenAI embeddings
│   ├── vector_store.py            # Pinecone management
│   ├── bm25_index.py             # Keyword search index
│   └── run_phase5.py             # Full pipeline
├── retrieval/          # ✅ Phase 4 COMPLETE
│   ├── hybrid_search.py           # Vector + BM25 + RRF fusion
│   ├── reranker.py               # Cohere semantic reranking
│   ├── context_organizer.py      # Result aggregation
│   ├── multi_step_retriever.py   # Full retrieval orchestrator
│   └── test_phase4.py            # Test suite
├── generation/         # ✅ Phase 6 COMPLETE
│   ├── answer_generator.py        # LLM generation (Groq)
│   ├── response_validator.py      # Quality validation
│   └── end_to_end_pipeline.py    # Complete RAG pipeline
├── evaluation/         # ✅ Phase 7 COMPLETE
│   ├── retrieval_metrics.py       # IR metrics (P, R, MRR, MAP, NDCG)
│   ├── generation_metrics.py      # NLG metrics
│   └── comprehensive_evaluation.py # Batch evaluation
└── utils/
    └── toc_filter.py              # TOC filtering utility
```

## Performance & Quality Metrics

### Chunk Quality (Phase 2)
- Overall score: 0.89/1.00 ✅ (target: >0.80)
- Structure preservation: 0.99/1.00
- Context completeness: 0.96/1.00

### Query Understanding (Phase 3)
- 100% test success rate on 5 complex queries
- Average 2.8 sub-questions per complex query

### Embeddings (Phase 5)
- 2,106 vectors in Pinecone (3072-dim)
- BM25 vocabulary: 16,460 terms
- 0% upload failures

### Retrieval Performance (Phase 4 + 7)
Based on evaluation results:
- **Precision@10**: 0.52-0.70 (target: >0.70)
- **Recall@10**: 0.41-0.64 (target: >0.60)
- **MRR**: 0.39-0.75 (target: >0.70)
- **Coverage**: 0.66-0.75 (target: >0.80)
- **Diversity**: 1.00 (target: >0.70) ✅

### Generation Quality (Phase 6 + 7)
Based on evaluation results:
- **Overall Score**: 0.88-0.92 (target: >0.75) ✅
- **Completeness**: 1.00 (target: >0.80) ✅
- **Word Count**: 400-600 words (ideal range)
- **Quality Distribution**: 90% excellent (≥0.85)

### Performance
- **Avg Time per Query**: 14-30 seconds
  - Query understanding: ~1-2s
  - Retrieval (3 sub-questions): ~8-15s
  - Generation: ~2-4s
  - Validation: <1s
- **Cost per Query**: ~$0.002 (OpenAI embeddings + Cohere reranking)
- **Groq LLM**: FREE (within rate limits)

## Common Patterns

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
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing {count} items")
logger.warning(f"Skipped {skip_count} items")
```

### Dataclasses
Extensively used for structured data:
```python
from dataclasses import dataclass, field, asdict

@dataclass
class ChunkMetadata:
    chunk_id: str
    page_start: int
    heading_path: List[str] = field(default_factory=list)
```

Use `asdict()` for JSON serialization.

### Embedding Generation Pattern
**Important**: Two separate methods for different use cases:
```python
generator = EmbeddingGenerator()

# For simple text strings (queries)
embeddings = generator.generate_embeddings(["query1", "query2"])

# For chunk dictionaries with metadata
embedded_chunks = generator.generate_embeddings_for_chunks(chunks, show_progress=True)
```

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Use `python -m src.module.name` instead of running files directly

### Pinecone Import Error
**Problem**: Error about `pinecone-client` vs `pinecone`
**Solution**: `pip uninstall pinecone-client -y && pip install -U pinecone`

### Settings AttributeError
**Problem**: `'Settings' object has no attribute 'paths'`
**Solution**: Settings are flat - use `settings.pdf_path` not `settings.paths.pdf_path`

### Groq Rate Limit Error
**Problem**: `Error code: 429 - Rate limit reached for model llama-3.3-70b-versatile`
**Solution**:
- Wait 10 minutes or until daily reset (midnight UTC)
- Spread testing across multiple days
- Use smaller test batches (5-10 queries)
- Upgrade to Groq Dev Tier for higher limits

### Evaluation Fails on Many Queries
**Problem**: Only first 1-2 queries succeed in batch evaluation
**Root Cause**: Groq free tier limit (100K tokens/day ≈ 14 queries)
**Solution**: Run evaluation in batches across multiple days:
```bash
# Day 1: First 10 queries
python -m src.evaluation.comprehensive_evaluation
# Enter: 10

# Day 2: Next 10 queries (manually edit test_queries.json)
# Day 3: Final 10 queries
```

## API Keys Required

All keys configured in `.env`:
1. **OpenAI**: https://platform.openai.com/api-keys
   - Used for: Embeddings (text-embedding-3-large)
   - Cost: ~$0.08 one-time for all chunks
   - Per-query cost: ~$0.0005 (3 sub-questions × 1 embedding each)

2. **Pinecone**: https://app.pinecone.io/
   - Used for: Vector database
   - Free tier: 100K vectors, 1 index
   - Current usage: 2,106 vectors

3. **Cohere**: https://dashboard.cohere.com/api-keys
   - Used for: Semantic reranking (rerank-english-v3.0)
   - Free tier: 1000 requests/month
   - Per-query cost: ~$0.0015 (3 sub-questions × 1 rerank each)

4. **Groq**: https://console.groq.com/keys
   - Used for: LLM inference (Llama 3.3 70B)
   - Free tier: 100K tokens/day ≈ 14 queries/day
   - Per-query usage: ~7,000 tokens
   - **CRITICAL**: Monitor usage to avoid rate limits

## Current State Summary

### ✅ Completed (78%)
- **Phase 1**: Foundation & Setup
- **Phase 2**: Document Processing (2,106 chunks, quality: 0.89)
- **Phase 3**: Query Understanding (decomposition + classification + intent)
- **Phase 4**: Multi-Step Retrieval (hybrid search + reranking + organization)
- **Phase 5**: Embeddings & Indexing (Pinecone + BM25)
- **Phase 6**: Advanced Generation (multi-strategy + validation)
- **Phase 7**: Evaluation & Testing (comprehensive metrics)

### 🚧 Pending (22%)
- **Phase 8**: UI Integration (Streamlit web app)
- **Phase 9**: Documentation & Deployment (Docker, API, deployment guide)

### 📊 System Capabilities (Current)
- ✅ Process 2,257-page PDFs with structure preservation
- ✅ Generate 2,106 context-aware chunks with 20+ metadata fields
- ✅ Understand complex multi-topic queries via LLM decomposition
- ✅ Hybrid retrieval (Vector + BM25 + RRF fusion + Cohere reranking)
- ✅ Multi-step retrieval with context chaining
- ✅ Strategy-aware answer generation (4 different strategies)
- ✅ Response validation and quality scoring
- ✅ Comprehensive evaluation framework (IR + NLG metrics)
- ⏳ Web UI (Phase 8 needed)
- ⏳ Production deployment (Phase 9 needed)

### 🎯 Next Priority
**Phase 8: UI Integration** - Build Streamlit web app for interactive query testing with real-time pipeline visualization and metrics display.

---

**See Also**:
- `README.md` - Project overview and architecture
- `PROGRESS.md` - Detailed phase breakdowns
- `PHASE_3_COMPLETE.md` - Query understanding details
- `PHASE_4_COMPLETE.md` - Multi-step retrieval details
- `PHASE_6_COMPLETE.md` - Advanced generation details
- `PHASE_7_COMPLETE.md` - Evaluation framework details
- `TOC_HANDLING.md` - TOC filtering strategy
