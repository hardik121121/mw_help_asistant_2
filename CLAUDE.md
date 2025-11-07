# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference (Most Critical Info)

### 🚀 Most Common Commands (90% of usage)

```bash
# 1. Launch the Streamlit UI (most common)
./run_app.sh

# 2. Run comprehensive evaluation (test changes)
python -m src.evaluation.comprehensive_evaluation

# 3. Test end-to-end pipeline (single query)
python -m src.generation.end_to_end_pipeline

# 4. Validate configuration
python -m config.settings

# 5. Reprocess documents (if PDF changes)
python -m src.ingestion.pymupdf_processor  # ~1 min (PRODUCTION)
python -m src.ingestion.hierarchical_chunker  # ~2 min

# 6. Rebuild indexes (if chunks change)
python -m src.database.run_phase5  # ~5 min
```

### 🔥 Critical Patterns (Most Common Errors)

**Module Execution**: ALWAYS use `python -m src.module.name` (never `python src/module/name.py`)

**Settings Access**: Flat structure (not nested)
- ✅ `settings.pdf_path`
- ❌ `settings.paths.pdf_path`

**Content Mapping**: Required for Pinecone retrieval
```python
content = self.chunk_content_map.get(chunk_id, '')  # Always load from map
```

**Query Expansion**: ALL queries are expanded before retrieval (automatic in `hybrid_search.py`)

**Rate Limits**: Groq free tier = ~14 queries/day. Always test with 5 queries first.

**Critical Files**:
- `docs/technical/architecture.md` - Complete architecture guide
- `config/settings.py` - All configuration (flat Pydantic model)
- `src/retrieval/hybrid_search.py` - Content/metadata mapping fix
- `src/query/query_expander.py` - 32 synonym mappings

## Project Overview

Maximum-quality RAG system for complex multi-topic queries across 2300+ pages. Implements **query decomposition + hierarchical chunking + multi-step retrieval + advanced generation**.

**Status**: Phases 1-8 complete (89%). Full RAG pipeline operational with Streamlit UI.

**🔴 CRITICAL: PDF Processor Used**: **PyMuPDF** (NOT Docling)
- Docling failed at page 495/2257 due to OCR errors
- Switched to PyMuPDF - completed all 2257 pages in ~1 minute
- File `cache/docling_processed.json` is **misleadingly named** - contains PyMuPDF output!
- **README.md mentions Docling** but production uses PyMuPDF - this is a known documentation lag

## Critical Development Rules

### 0. Python Module Execution (MOST FUNDAMENTAL)

**NEVER run Python files directly. ALWAYS use module syntax.**

```bash
# ❌ WRONG - Will cause import errors
python src/query/query_decomposer.py
python3 src/retrieval/hybrid_search.py

# ✅ CORRECT - Works from project root
python -m src.query.query_decomposer
python -m src.retrieval.hybrid_search
python -m src.evaluation.comprehensive_evaluation

# Why: Module syntax ensures Python includes project root in sys.path
# This allows imports like "from config.settings import get_settings" to work
```

**This is THE most common error. When in doubt, use `python -m`.**

### 1. PDF Processing Strategy (MOST CRITICAL)

**🔴 PRODUCTION USES PyMuPDF, NOT DOCLING!**

**What happened**: Docling failed at 22% (page 495), switched to PyMuPDF which succeeded in 1 min.

**PyMuPDF Heading Detection** (font-based, no ML):
```python
heading_1_size: 20  # Font ≥20pt → H1
heading_2_size: 16  # Font ≥16pt → H2
heading_3_size: 14  # Font ≥14pt → H3
heading_4_size: 12  # Font ≥12pt + bold → H4
```

**Evidence in metadata**:
```json
"font_size": 8.15999984741211,
"font_name": "LiberationSerif",
"is_bold": false
```
This is PyMuPDF's signature, NOT Docling's!

### 2. Critical Pinecone Metadata Fix (Nov 2, 2024)

Pinecone has 40KB metadata limit. Vector search was returning empty content.

**The Fix** (in `hybrid_search.py`):
```python
# Load full content/metadata at initialization
self.chunk_content_map = {chunk['metadata']['chunk_id']: chunk.get('content', '')}
self.chunk_metadata_map = {chunk['metadata']['chunk_id']: chunk.get('metadata', {})}

# During retrieval: restore full data
content = self.chunk_content_map.get(chunk_id, '')
full_metadata = self.chunk_metadata_map.get(chunk_id, {})
merged_metadata = {**match.metadata, **full_metadata}
```

### 3. Groq Rate Limits (CRITICAL)

**Free tier**: 100K tokens/day ≈ 14 queries/day
- Query understanding: ~1K tokens
- Answer generation: ~6K tokens
- **Total per query**: ~7K tokens

**Best practice**: Run evaluations in batches of 5 queries across multiple days.

## Common Commands

### Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m config.settings  # Validate
```

### Running the System
```bash
# Launch Streamlit UI
./run_app.sh

# Test end-to-end pipeline
python -m src.generation.end_to_end_pipeline

# Run evaluation (5 queries recommended)
python -m src.evaluation.comprehensive_evaluation
```

### Testing Individual Components
```bash
# Query Understanding Components
python -m src.query.query_decomposer      # Test query decomposition
python -m src.query.query_classifier      # Test query classification
python -m src.query.query_expander        # Test query expansion
python -m src.query.intent_analyzer       # Test intent analysis

# Retrieval Components
python -m src.retrieval.hybrid_search     # Test hybrid search
python -m src.retrieval.multi_step_retriever  # Test multi-step retrieval

# Generation Components
python -m src.generation.answer_generator # Test answer generation
python -m src.generation.response_validator  # Test validation

# Database Components
python -m src.database.bm25_index         # Test BM25 index
python -m src.database.vector_store       # Test Pinecone operations
```

**Note**: All modules with `if __name__ == "__main__"` can be tested independently. ALWAYS use `python -m` syntax, never run files directly.

## Architecture Overview

```
PDF → PyMuPDF → Hierarchical Chunks → Embeddings → Pinecone + BM25
                                                          ↓
Query → Decomposition → Multi-Step Retrieval → Generation → Answer
```

**Module Organization**:
- `src/ingestion/` - PDF processing, chunking, quality evaluation
- `src/query/` - Decomposition, classification, expansion, intent analysis
- `src/database/` - Embeddings, Pinecone, BM25 indexing
- `src/retrieval/` - Hybrid search, reranking, context organization
- `src/generation/` - Answer generation, validation, end-to-end pipeline
- `src/evaluation/` - Metrics calculation, comprehensive evaluation
- `src/utils/` - TOC filtering, helpers
- `config/` - Settings (Pydantic-based validation)
- `scripts/` - Standalone utilities (compare, diagnose, enrich)

**Key Innovations**:
1. **Hierarchical Context**: Every chunk includes full section hierarchy
2. **Rich Metadata**: 20+ fields per chunk (location, hierarchy, content flags)
3. **Multi-Step Retrieval**: Independent retrieval per sub-question, then deduplicate
4. **Query Expansion**: 32 synonym mappings, automatically applied
5. **Context Chaining**: Earlier sub-question results enrich later ones
6. **Strategy-Aware Generation**: 4 different generation strategies

**For complete architecture**: See `docs/technical/architecture.md`

## Data Flow & Files

**Inputs**:
- `data/helpdocs.pdf` (157 MB, 2257 pages)
- `.env` (API keys)
- `tests/test_queries.json` (30 test queries)

**Phase 2 Outputs**:
- `cache/docling_processed.json` (43 MB) - **PyMuPDF output despite name!**
- `cache/hierarchical_chunks_filtered.json` (4.5 MB, 2,106 chunks)
- `cache/images/` (1,454 images)

**Phase 5 Outputs**:
- `cache/hierarchical_embeddings.pkl` (59 MB)
- `cache/bm25_index.pkl` (64 MB)
- Pinecone index: `watermelon-docs-v2` (2,106 vectors)

## Performance Metrics (Latest - Nov 7, 2024)

**After Query Expansion + RRF Tuning** (5 queries evaluated):
- Precision@10: **0.740** (+11% from 50/50 baseline)
- Recall@10: **0.561** (-12% trade-off for precision)
- MRR: **1.000** (perfect - best result always ranked #1!) 🎯
- Coverage: **0.833** (+13%)

**RRF Weights** (tuned for technical documentation):
- Vector (semantic): 45%
- BM25 (keyword): 55%
- **Why**: Exact keywords (integration names, features) matter more than pure semantics

**Generation**:
- Overall Score: **0.912** (target >0.75) ✅
- Completeness: **1.000** (perfect)
- Avg Word Count: 421

**Performance**:
- Avg time: **28.0s** per query
- Cost: **$0.003** per query

**See**: `docs/evaluation/final-results.md` for complete analysis.

## Utility Scripts

Located in `scripts/` directory - all standalone, no src/ imports required:

```bash
# Compare evaluation results (A/B testing)
python scripts/compare_evaluations.py tests/results/baseline.json tests/results/new.json

# Diagnose quality issues (empty content, missing images)
python scripts/diagnose_quality.py

# Enrich chunks with computed metadata
python scripts/enrich_chunks.py

# Run quality improvement test suite
./scripts/test_quality_improvement.sh
```

## Development Workflows

### Debugging Pipeline Issues

When the end-to-end pipeline fails or produces poor results:

```bash
# 1. Test each stage independently
python -m src.query.query_understanding  # Check decomposition
python -m src.retrieval.hybrid_search    # Check retrieval
python -m src.generation.answer_generator  # Check generation

# 2. Enable debug logging
# In .env: LOG_LEVEL=DEBUG ENABLE_DEBUG_MODE=true

# 3. Check intermediate outputs
python -c "from dataclasses import asdict; import json;
from src.generation.end_to_end_pipeline import EndToEndPipeline;
pipeline = EndToEndPipeline();
result = pipeline.process('test query');
print(json.dumps(asdict(result), indent=2, default=str))"

# 4. Inspect retrieval quality
python scripts/diagnose_quality.py
```

### Adding New Integration Synonyms
```python
# src/query/query_expander.py
self.integration_aliases = {
    'zendesk': ['zendesk support', 'zendesk help desk'],
    'hubspot': ['hubspot crm', 'hubspot marketing'],
}
```

### Incremental Evaluation (RECOMMENDED)
```bash
# Day 1: Baseline (5 queries)
python -m src.evaluation.comprehensive_evaluation
# Enter: 5
cp tests/results/comprehensive_evaluation.json tests/results/baseline.json

# Day 2: After changes (5 queries)
python -m src.evaluation.comprehensive_evaluation
# Enter: 5

# Compare results
python scripts/compare_evaluations.py tests/results/baseline.json tests/results/comprehensive_evaluation.json
```

### Working with Content Mapping
```python
# Required pattern when retrieving from Pinecone:
chunk_id = match.metadata.get('chunk_id')
content = self.chunk_content_map.get(chunk_id, '')  # Full content
full_metadata = self.chunk_metadata_map.get(chunk_id, {})
merged_metadata = {**match.metadata, **full_metadata}

result = {'content': content, 'metadata': merged_metadata, 'score': match.score}
```

## Architecture Patterns

### Dataclass-Based Architecture

**All major objects are dataclasses** (no ORM/database models):
- `QueryUnderstanding`, `DecomposedQuery`, `SubQuestion`
- `RetrievalResult`, `OrganizedContext`
- `GeneratedAnswer`, `ValidationResult`, `PipelineResult`

**Critical patterns**:
```python
# ✅ Nested access
pipeline_result.answer.answer
pipeline_result.validation.overall_score

# ❌ Flattening assumptions
pipeline_result.answer_text  # AttributeError!

# ✅ JSON serialization
from dataclasses import asdict
result_dict = asdict(result)
json.dump(result_dict, f)

# ✅ Mutable defaults
@dataclass
class MyClass:
    items: List[str] = field(default_factory=list)  # Correct
    tags: List[str] = []  # Wrong - shared between instances!
```

**Debugging**:
```python
# ✅ Best
from dataclasses import asdict
import json
print(json.dumps(asdict(pipeline_result), indent=2))

# Or access specific fields
print(f"Answer: {pipeline_result.answer.answer}")
print(f"Score: {pipeline_result.validation.overall_score}")
```

### Synchronous Architecture

**Entire codebase is synchronous** - no `async`/`await`:
- Simpler debugging
- Context chaining requires sequential processing
- Trade-off: simplicity vs speed (parallelization possible in Phase 9)

### RRF Parameters (TUNED - Nov 7, 2024)

```python
# In config/settings.py (configurable)
rrf_k = 60  # Standard value (range: 20-100)
vector_weight = 0.45  # 45% semantic (TUNED from 0.5)
bm25_weight = 0.55    # 55% keyword (TUNED from 0.5)
```

**Why 45/55 split:**
- Technical docs need exact keyword matching (integration names, features)
- Achieved: MRR 100% (perfect ranking), +11% precision
- Trade-off: -12% recall (acceptable for better UX)

**How to tune:**
1. Run baseline: `python -m src.evaluation.comprehensive_evaluation` (5 queries)
2. Modify `vector_weight` and `bm25_weight` in `config/settings.py`
3. Re-run evaluation
4. Compare with: `python scripts/compare_evaluations.py tests/results/baseline.json tests/results/comprehensive_evaluation.json`

## Troubleshooting

### Pipeline Runs Slowly
**Problem**: End-to-end queries take >45 seconds
**Solutions**:
1. Reduce `vector_top_k` and `bm25_top_k` from 50 to 30
2. Reduce `rerank_top_k` from 20 to 10
3. Disable context chaining: `enable_context_chaining=False`
4. Check Groq rate limits (free tier has throttling)

### Poor Retrieval Results
**Problem**: Retrieved chunks don't answer the query
**Solutions**:
1. Check query expansion: `python -m src.query.query_expander`
2. Verify BM25 index is loaded: Check `cache/bm25_index.pkl` size (should be ~64MB)
3. Test vector search alone: Set `bm25_weight=0.0, vector_weight=1.0`
4. Test keyword search alone: Set `vector_weight=0.0, bm25_weight=1.0`
5. Run diagnostics: `python scripts/diagnose_quality.py`

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**: Use `python -m src.module.name` not `python src/module/name.py`

### Pinecone Import Error
**Solution**: `pip uninstall pinecone-client -y && pip install -U pinecone`

### Settings AttributeError
**Solution**: Settings are flat - `settings.pdf_path` not `settings.paths.pdf_path`

### Groq Rate Limit
**Solution**: Wait 10 min or until midnight UTC. Use batches of 5 queries.

### Empty Retrieval Content
**Problem**: Chunks have 0 chars, no images
**Solution**: Check content_map is loaded (see "Working with Content Mapping" above)

## API Keys Required

1. **OpenAI**: Embeddings (~$0.08 one-time, ~$0.0005 per query)
2. **Pinecone**: Vector DB (free tier: 100K vectors)
3. **Cohere**: Reranking (free: 1000 requests/month)
4. **Groq**: LLM (free: 100K tokens/day ≈ 14 queries)

Get keys at: `docs/setup/api-keys.md`

## Summary: Key Architectural Insights

**Most critical non-obvious patterns**:

1. **🔴 PyMuPDF is Production, NOT Docling** - File `docling_processed.json` is misleading!
2. **Query Expansion is Automatic** - Every query → 3 variations (32 synonym mappings)
3. **Dataclasses Everywhere** - No ORM, just dataclasses + pickle/JSON
4. **Three-Map Pinecone Recovery** - content_map + metadata_map + embeddings (40KB limit workaround)
5. **Synchronous by Design** - No async/await (intentional simplicity)
6. **Context Chaining** - Sequential processing required (can't parallelize)
7. **RRF k=60, 70/30 Split** - Tunable hybrid search parameters
8. **All Modules Runnable** - Test independently with `if __name__ == "__main__"`
9. **Font-Based Detection** - 20pt=H1, 16pt=H2, 14pt=H3, 12pt+bold=H4

**Most Common Errors to Avoid**:
- ❌ Assuming Docling is used (it's PyMuPDF!)
- ❌ Running files directly instead of `python -m src.module.name`
- ❌ Forgetting content_map when retrieving from Pinecone
- ❌ Assuming nested settings (they're flat)
- ❌ JSON serializing dataclasses without `asdict()`
- ❌ Using `= []` for mutable defaults instead of `field(default_factory=list)`
- ❌ Evaluating all 30 queries at once (exceeds Groq limits - use 5!)
- ❌ Not saving baseline before changes (use `compare_evaluations.py`)
- ❌ Trusting README.md over CLAUDE.md (README has outdated Docling references)

## Critical File Naming Gotchas

| File Name | Actual Content | Why Misleading |
|-----------|----------------|----------------|
| `cache/docling_processed.json` | **PyMuPDF output** | Named before switching processors |
| `src/ingestion/docling_processor.py` | **Unused/broken** | Kept for reference only |

**Verify**: Check metadata for `"font_size"` (PyMuPDF) vs ML labels (Docling)

## Documentation Structure

- `docs/technical/architecture.md` - **Complete system architecture**
- `docs/evaluation/final-results.md` - Performance metrics & analysis
- `docs/guides/quick-start-ui.md` - Streamlit interface guide
- `docs/guides/quality-improvement.md` - Troubleshooting output quality
- `docs/setup/getting-started.md` - Comprehensive setup guide

## Quick Wins for Phase 9

**Performance** (40-50% speedup):
- Parallelize INDEPENDENT sub-questions
- Redis caching for query results
- Async processing for non-critical ops

**Quality** (+10-15%):
- Increase top_k from 50 → 75
- Fine-tune embedding model
- Cross-encoder reranking

---

**Built for Maximum Quality. Designed for Complex Queries. Optimized for Production.**
