# Quick Setup Guide

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd /home/hardik121/wm_help_assistant_2

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note**: Docling installation may take 5-10 minutes as it has many dependencies.

---

### 2. Configure API Keys

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your API keys
nano .env
# OR
code .env  # If using VS Code
```

**Required API Keys**:
1. **OpenAI** (for embeddings & query decomposition): https://platform.openai.com/api-keys
2. **Pinecone** (for vector database): https://app.pinecone.io/
3. **Cohere** (for re-ranking): https://dashboard.cohere.com/api-keys
4. **Groq** (for LLM): https://console.groq.com/keys

**Cost Estimate**:
- One-time setup: ~$3-5 (OpenAI embeddings)
- Per query: ~$0.002-0.005
- Monthly (300 queries): ~$10-15

---

### 3. Validate Configuration

```bash
# Test configuration loading
python config/settings.py
```

You should see:
```
📋 CONFIGURATION SUMMARY
========================
🔑 API Configuration:
   OpenAI: ✓ Set
   Pinecone: ✓ Set
   Cohere: ✓ Set
   Groq: ✓ Set
...
✅ Configuration is valid!
```

---

## 📄 Process the PDF (Phase 2 Components)

### Step 1: Extract PDF Structure with Docling

```bash
python src/ingestion/docling_processor.py
```

**Output**:
- `cache/docling_processed.json` (23-25 MB) - Structured document data
- `cache/images/` - Extracted PNG images
- Processing time: ~10-20 minutes for 2257 pages

**What it does**:
- Extracts text with heading hierarchy
- Identifies tables and exports as HTML/Markdown
- Extracts images with captions
- Builds table of contents
- Preserves bounding boxes for all elements

---

### Step 2: Create Hierarchical Chunks

```bash
python src/ingestion/hierarchical_chunker.py
```

**Output**:
- `cache/hierarchical_chunks.json` (3-5 MB) - Chunked document with metadata
- Processing time: ~1-2 minutes

**What it does**:
- Groups content by sections
- Adds hierarchical context to each chunk
- Includes image/table references
- Classifies content type and technical depth
- Creates ~2000-3000 chunks with rich metadata

---

### Step 3: Evaluate Chunk Quality

```bash
python src/ingestion/chunk_evaluator.py
```

**Output**:
- `tests/results/chunk_quality_report.txt` - Comprehensive quality report
- `tests/results/chunk_quality_report.json` - Metrics in JSON format
- Processing time: <1 minute

**What it shows**:
- Size consistency score
- Structure preservation score
- Context completeness score
- Overall quality score (target: >0.80)
- Problematic chunks identified

---

## 🧪 Test Queries

The project includes 30 complex test queries in `tests/test_queries.json`:

**Example Queries**:
1. "How do I create a no-code block on Watermelon platform and process it for Autonomous Functional Testing?"
2. "What are the integration steps for MS Teams and how do I configure automated responses with chatbots?"
3. "How do I set up WhatsApp Business API integration, create conversation flows, and implement message templates?"

These are designed to test multi-topic understanding.

---

## 📊 Phase 2 Pipeline Overview

```
PDF (157 MB)
    ↓
[Docling Processor]  ← Phase 2.1
    ↓
Structured JSON (23 MB) + Images
    ↓
[Hierarchical Chunker]  ← Phase 2.2
    ↓
Chunks with Context (3 MB)
    ↓
[Quality Evaluator]  ← Phase 2.3
    ↓
Quality Report + Metrics
```

---

## ⚡ Quick Commands Reference

### Configuration
```bash
# Validate configuration
python config/settings.py

# Check configuration in Python
python -c "from config.settings import get_settings; get_settings().print_summary()"
```

### Processing
```bash
# Full pipeline (Step 1-3)
python src/ingestion/docling_processor.py && \
python src/ingestion/hierarchical_chunker.py && \
python src/ingestion/chunk_evaluator.py
```

### View Test Queries
```bash
# Pretty print test queries
python -c "import json; print(json.dumps(json.load(open('tests/test_queries.json')), indent=2))" | less
```

### Check Cache
```bash
# List cache contents
ls -lh cache/
ls -lh cache/images/ | wc -l  # Count images
```

---

## 🔍 Troubleshooting

### Issue: Docling import error
**Solution**:
```bash
# Reinstall with specific version
pip install docling>=1.0.0

# Or try development version
pip install git+https://github.com/DS4SD/docling.git
```

### Issue: PDF not found
**Solution**:
```bash
# Check PDF location
ls -lh data/helpdocs.pdf

# If missing, copy from parent directory
cp /home/hardik121/wm_help_assistant_2/helpdocs.pdf data/
```

### Issue: API key validation failed
**Solution**:
```bash
# Check .env file exists
cat .env | grep API_KEY

# Ensure no placeholder values (xxx...)
# Get real API keys from the services
```

### Issue: Out of memory during processing
**Solution**:
```python
# In docling_processor.py, reduce batch size
# Or process in chunks by page ranges
```

---

## 📁 Directory Structure

```
wm_help_assistant_2/
├── config/
│   └── settings.py              # Configuration management
├── src/
│   ├── ingestion/
│   │   ├── docling_processor.py  # PDF → Structure
│   │   ├── hierarchical_chunker.py  # Structure → Chunks
│   │   └── chunk_evaluator.py    # Quality evaluation
│   ├── database/                # (Phase 5)
│   ├── retrieval/               # (Phase 4)
│   ├── generation/              # (Phase 6)
│   ├── query/                   # (Phase 3)
│   ├── memory/                  # (Phase 8)
│   └── utils/
├── tests/
│   ├── test_queries.json        # 30 complex test queries
│   └── results/                 # Evaluation outputs
├── data/
│   └── helpdocs.pdf             # Source PDF (157 MB)
├── cache/
│   ├── docling_processed.json   # Structured document
│   ├── hierarchical_chunks.json # Chunks with metadata
│   └── images/                  # Extracted images
├── requirements.txt
├── .env.example
├── .env                         # Your API keys (create this)
├── PROGRESS.md                  # Development progress
└── SETUP.md                     # This file
```

---

## 🎯 Next Steps

After completing Phase 2 setup:

1. **Phase 3**: Query Understanding Engine
   - Query decomposition
   - Intent classification
   - Sub-question generation

2. **Phase 4**: Multi-Step Retrieval
   - Hybrid search per sub-question
   - Context organization
   - Diversity ranking

3. **Phase 5**: Embeddings & Indexing
   - Generate embeddings
   - Create Pinecone index
   - Build BM25 index

4. **Phases 6-9**: Generation, Evaluation, UI, Deployment

---

## 💡 Tips

1. **Save processing outputs**: Don't delete cache files - regenerating takes time
2. **Test incrementally**: Run each component separately to catch issues early
3. **Monitor costs**: Track API usage during development
4. **Use free tiers**: Groq (14.4K req/day), Pinecone (100K vectors), Cohere (1K calls/month)
5. **Check quality scores**: Target >0.80 for overall chunk quality

---

## 📚 Additional Resources

- **Docling Documentation**: https://github.com/DS4SD/docling
- **LangChain Docs**: https://python.langchain.com/docs/get_started/introduction
- **Pinecone Docs**: https://docs.pinecone.io/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings

---

## ✅ Validation Checklist

Before proceeding to Phase 3:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip list | grep docling`)
- [ ] .env file created with real API keys
- [ ] Configuration validated (python config/settings.py)
- [ ] PDF exists at `data/helpdocs.pdf`
- [ ] Docling processor runs successfully
- [ ] Hierarchical chunks created
- [ ] Quality report generated
- [ ] Quality score >0.80 (if not, review settings)

---

**Need Help?** Check `PROGRESS.md` for detailed technical information about each component.
