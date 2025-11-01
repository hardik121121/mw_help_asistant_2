# Watermelon Documentation Assistant 🤖

> **Maximum-Quality RAG System for Complex Multi-Topic Queries**

A production-grade Retrieval-Augmented Generation (RAG) system designed to handle complex queries across 2300+ pages of documentation. Built with hierarchical chunking, query decomposition, and multi-step retrieval to answer questions that span multiple topics.

---

## 🎯 Problem Statement

Traditional RAG systems struggle with complex queries like:
- *"How do I create a no-code block on Watermelon and process it for Autonomous Functional Testing?"*
- *"What are the integration steps for MS Teams and how do I configure automated responses?"*

These questions require:
1. Understanding multiple topics simultaneously
2. Retrieving context from different document sections
3. Integrating information across topics
4. Providing step-by-step, comprehensive answers

---

## 💡 Our Solution

### Key Innovations

#### 1. **Hierarchical Document Processing**
- Uses **Docling** (not PyMuPDF) to preserve document structure
- Maintains heading hierarchy (H1→H2→H3→H4)
- Extracts tables and images with context
- Preserves cross-references and semantic boundaries

#### 2. **Context-Aware Chunking**
- Section-based chunking respects heading boundaries
- **Context injection**: Each chunk gets section hierarchy prepended
- Multi-page topic handling merges related content
- **20+ metadata fields** per chunk for smart retrieval

#### 3. **Query Decomposition**
```
Complex Query → 2-4 Sub-Questions → Multi-Step Retrieval → Integrated Answer
```
- LLM-based query analysis
- Dependency detection (sequential vs parallel)
- Query expansion with synonyms

#### 4. **Multi-Step Retrieval**
- **Hybrid search** per sub-question (Vector + BM25)
- **Reciprocal Rank Fusion** (RRF) combines results
- **Cohere Re-ranking** for precision
- **Context chaining** between retrieval steps

#### 5. **Advanced Generation**
- Multi-context prompting
- Response validation
- Smart image selection
- Per-section citations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUERY                             │
│  "How do I create a no-code block and use it for testing?" │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               QUERY UNDERSTANDING (Phase 3)                 │
│  • Decomposition: 4 sub-questions                           │
│  • Classification: multi-topic_procedural                   │
│  • Intent: Create + Configure + Integrate                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             MULTI-STEP RETRIEVAL (Phase 4)                  │
│  For each sub-question:                                     │
│    1. Vector Search (top-30)                                │
│    2. BM25 Search (top-30)                                  │
│    3. RRF Fusion                                            │
│    4. Cohere Rerank (top-10)                                │
│  → Combine, deduplicate, organize by topic                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             CONTEXT ORGANIZATION (Phase 4)                  │
│  • Topic clustering                                         │
│  • Chronological ordering                                   │
│  • Relationship mapping                                     │
│  → 15-20 relevant chunks with images/tables                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│            ADVANCED GENERATION (Phase 6)                    │
│  • Multi-topic prompt engineering                           │
│  • Step-by-step reasoning                                   │
│  • Response validation                                      │
│  • Citations & images                                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                COMPREHENSIVE ANSWER                         │
│  ✓ All sub-topics addressed                                │
│  ✓ Step-by-step instructions                               │
│  ✓ Proper formatting                                       │
│  ✓ Citations by section                                    │
│  ✓ Relevant images                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/navigate to project
cd /home/hardik121/wm_help_assistant_2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Add your API keys
nano .env
```

**Required API Keys**:
- OpenAI (embeddings): https://platform.openai.com/api-keys
- Pinecone (vector DB): https://app.pinecone.io/
- Cohere (re-ranking): https://dashboard.cohere.com/api-keys
- Groq (LLM): https://console.groq.com/keys

### 3. Process Documentation

```bash
# Step 1: Extract structure with Docling (~15 min)
python src/ingestion/docling_processor.py

# Step 2: Create hierarchical chunks (~2 min)
python src/ingestion/hierarchical_chunker.py

# Step 3: Evaluate quality (<1 min)
python src/ingestion/chunk_evaluator.py
```

### 4. Run Application (Coming in Phase 8)

```bash
streamlit run app.py
```

**See `SETUP.md` for detailed instructions.**

---

## 📊 Current Progress

| Phase | Status | Completion |
|-------|--------|------------|
| **Phase 1**: Foundation & Setup | ✅ Complete | 100% |
| **Phase 2**: Advanced Document Processing | ✅ Complete | 100% |
| **Phase 3**: Query Understanding Engine | 🚧 Pending | 0% |
| **Phase 4**: Multi-Step Retrieval System | 🚧 Pending | 0% |
| **Phase 5**: Embeddings & Indexing | 🚧 Pending | 0% |
| **Phase 6**: Advanced Generation Pipeline | 🚧 Pending | 0% |
| **Phase 7**: Evaluation & Testing | 🚧 Pending | 0% |
| **Phase 8**: UI Integration & Polish | 🚧 Pending | 0% |
| **Phase 9**: Documentation & Deployment | 🚧 Pending | 0% |

**Overall: 22% Complete (2/9 phases)**

**See `PROGRESS.md` for detailed progress tracking.**

---

## 🎨 Key Features

### ✅ Implemented (Phases 1-2)

#### Configuration System
- Pydantic-based validation
- Environment variable management
- Multi-section configuration
- Built-in error reporting

#### Docling PDF Processor
- Hierarchical structure extraction
- Table extraction (HTML/Markdown)
- Image extraction with captions
- Bounding box preservation
- Table of contents generation

#### Hierarchical Chunker
- Section-based chunking
- Context injection (heading path prepended)
- Multi-page topic merging
- 20+ metadata fields per chunk
- Content type classification
- Technical depth estimation

#### Quality Evaluation
- Size consistency scoring
- Structure preservation scoring
- Context completeness scoring
- Boundary analysis
- Problematic chunk detection
- Comprehensive reporting

### 🚧 Planned (Phases 3-9)

- Query decomposition & intent understanding
- Multi-step retrieval with context chaining
- Advanced re-ranking & diversity
- Multi-context generation
- Response validation
- Comprehensive evaluation suite
- Streamlit UI with debug features
- Docker deployment

---

## 🧪 Test Dataset

30 complex test queries in `tests/test_queries.json`:

**Example**:
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

**Query Types**:
- Multi-topic procedural
- Multi-topic integration
- Conceptual + procedural
- Troubleshooting
- Security & compliance

---

## 💾 Data Flow

### Document Processing Pipeline

```
PDF (157 MB, 2257 pages)
    ↓
┌────────────────────────────┐
│  Docling Processor         │
│  • Structure extraction    │
│  • Heading hierarchy       │
│  • Table/image extraction  │
└────────────────────────────┘
    ↓
Structured JSON (23 MB) + Images (~68 KB)
    ↓
┌────────────────────────────┐
│  Hierarchical Chunker      │
│  • Section grouping        │
│  • Context injection       │
│  • Metadata enrichment     │
└────────────────────────────┘
    ↓
Chunks (3 MB, ~2500 chunks)
    ↓
┌────────────────────────────┐
│  Embedding Generator       │
│  • OpenAI text-embedding-3 │
│  • 3072 dimensions         │
└────────────────────────────┘
    ↓
Embeddings (63 MB) + Pinecone Index
```

### Query Processing Pipeline (Planned)

```
User Query
    ↓
Query Decomposer → 2-4 Sub-Questions
    ↓
Multi-Step Retriever → Per-Question Results
    ↓
Context Organizer → Integrated Context
    ↓
LLM Generator → Comprehensive Answer
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Docling** - PDF processing with structure preservation
- **LangChain** - Text splitting & document processing
- **OpenAI** - Embeddings (text-embedding-3-large) & query decomposition
- **Pinecone** - Vector database (serverless, 3072-dim, cosine)
- **Cohere** - Re-ranking (rerank-english-v3.0)
- **Groq** - LLM inference (Llama 3.3 70B)
- **Streamlit** - Web UI

### Supporting Libraries
- **Pydantic** - Configuration validation
- **tiktoken** - Token counting
- **rank-bm25** - Keyword search
- **Pillow** - Image processing
- **loguru** - Logging
- **tenacity** - Retry logic

---

## 📈 Expected Performance

### Retrieval Quality (Targets)
- **Precision@10**: >0.85
- **MRR**: >0.7
- **Coverage**: >90% of topics in complex queries

### Generation Quality (Targets)
- **Completeness**: >90% of sub-questions answered
- **Accuracy**: >95% factually correct
- **Formatting**: >95% proper structure
- **Citations**: >95% claims cited

### Performance (Targets)
- Simple queries: <5s
- Complex queries: <10s
- Cost per query: <$0.01

### User Experience (Targets)
- Query success rate: >90%
- Response clarity: >85%
- Image relevance: >90%

---

## 💰 Cost Estimation

### One-Time Setup
- OpenAI embeddings (~2500 chunks): **$3-5**

### Per Query
- OpenAI query embedding: $0.0001
- Cohere re-ranking: $0.002
- Groq LLM: $0 (free tier)
- **Total per query**: ~$0.002-0.005

### Monthly (300 queries)
- ~**$10-15**

**Free Tier Limits**:
- Groq: 14,400 requests/day
- Pinecone: 100,000 vectors
- Cohere: 1,000 calls/month (then $0.002/call)

---

## 📁 Project Structure

```
wm_help_assistant_2/
├── config/
│   └── settings.py              # ✅ Configuration management
├── src/
│   ├── ingestion/
│   │   ├── docling_processor.py   # ✅ Docling-based PDF processing
│   │   ├── hierarchical_chunker.py # ✅ Context-aware chunking
│   │   └── chunk_evaluator.py     # ✅ Quality evaluation
│   ├── query/                     # 🚧 Query understanding (Phase 3)
│   ├── retrieval/                 # 🚧 Multi-step retrieval (Phase 4)
│   ├── generation/                # 🚧 Advanced generation (Phase 6)
│   ├── database/                  # 🚧 Vector DB (Phase 5)
│   ├── memory/                    # 🚧 Conversation (Phase 8)
│   └── utils/
├── tests/
│   ├── test_queries.json          # ✅ 30 complex test queries
│   └── results/                   # Evaluation outputs
├── data/
│   └── helpdocs.pdf               # ✅ Source PDF (157 MB)
├── cache/
│   ├── docling_processed.json     # Generated by Phase 2
│   ├── hierarchical_chunks.json   # Generated by Phase 2
│   └── images/                    # Extracted images
├── requirements.txt               # ✅ All dependencies
├── .env.example                   # ✅ Configuration template
├── PROGRESS.md                    # ✅ Development progress
├── SETUP.md                       # ✅ Setup instructions
└── README.md                      # This file
```

---

## 🔬 Evaluation Framework

### Chunk Quality Metrics
- Size consistency
- Structure preservation
- Context completeness
- Boundary analysis
- **Overall quality score**: Target >0.80

### Retrieval Metrics (Planned)
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Coverage (% topics retrieved)

### Generation Metrics (Planned)
- Completeness (all sub-topics addressed)
- Factual accuracy
- Formatting quality
- Citation accuracy
- Coherence

---

## 🤝 Contributing

This project follows a phased development approach:
1. Complete current phase
2. Evaluate quality metrics
3. Iterate if needed
4. Move to next phase

**Current Phase**: 3 (Query Understanding Engine)

---

## 📚 Documentation

- **`SETUP.md`** - Installation and configuration guide
- **`PROGRESS.md`** - Detailed progress tracking and technical details
- **`tests/test_queries.json`** - Test dataset with 30 complex queries
- **Code Documentation** - Comprehensive docstrings throughout

---

## 🎯 Success Criteria

### Phase 2 (Current)
- [x] Docling extracts structure correctly
- [x] Chunks preserve heading hierarchy
- [x] Quality score >0.80
- [x] Metadata includes images/tables
- [x] Evaluation framework working

### Final System
- [ ] Handles 90%+ of complex queries successfully
- [ ] Retrieval precision >0.85
- [ ] Generation accuracy >0.95
- [ ] Response time <10s for complex queries
- [ ] User satisfaction >85%

---

## 🐛 Known Issues & Limitations

### Current (Phase 2)
- Docling installation can be complex (many dependencies)
- PDF processing takes 10-20 minutes
- Quality highly depends on PDF structure

### Planned Solutions
- Docker container for easy setup (Phase 9)
- Incremental processing for large documents
- Fallback to PyMuPDF if Docling fails

---

## 📝 License

[Specify your license here]

---

## 👥 Authors

[Your team/name here]

---

## 🙏 Acknowledgments

- **Docling** team for structure-aware PDF processing
- **LangChain** community for RAG foundations
- **OpenAI**, **Pinecone**, **Cohere**, **Groq** for excellent APIs

---

## 📞 Support

For issues, questions, or contributions:
- Check `SETUP.md` for setup help
- Review `PROGRESS.md` for technical details
- Consult code documentation

---

**Last Updated**: 2024-11-01
**Version**: 0.2.0 (Phase 2 Complete)
**Next Milestone**: Phase 3 - Query Understanding Engine

---

## 🌟 Why This Approach is Better

### vs Traditional RAG
- ❌ Traditional: Flat chunks, lost context, single retrieval
- ✅ Ours: Hierarchical chunks, preserved context, multi-step retrieval

### vs Simple Chunking
- ❌ Simple: Arbitrary boundaries, no metadata, token-based
- ✅ Ours: Section-based, 20+ metadata fields, context-aware

### vs Single-Question Systems
- ❌ Single: Can't handle complex multi-topic queries
- ✅ Ours: Decomposes, retrieves per topic, integrates answers

---

**Built for Maximum Quality. Designed for Complex Queries. Optimized for Production.**
