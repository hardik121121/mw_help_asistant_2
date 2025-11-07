# Documentation Index

This directory contains all comprehensive documentation for the Watermelon Documentation Assistant RAG system.

## 📚 Quick Links

- **New Users**: Start with [Setup → Getting Started](setup/getting-started.md)
- **UI Guide**: See [Guides → Quick Start UI](guides/quick-start-ui.md)
- **API Keys**: See [Setup → API Keys](setup/api-keys.md)
- **Quality Issues**: See [Guides → Quality Improvement](guides/quality-improvement.md)

## 📁 Documentation Structure

### Setup & Installation

- **[Getting Started](setup/getting-started.md)** - Comprehensive setup guide with all phases
- **[Setup](setup/setup.md)** - Basic setup instructions
- **[API Keys](setup/api-keys.md)** - How to obtain required API keys (OpenAI, Pinecone, Cohere, Groq)

### Guides

- **[Quick Start UI](guides/quick-start-ui.md)** - How to use the Streamlit web interface
- **[Quality Improvement](guides/quality-improvement.md)** - Systematic guide for diagnosing and fixing output quality issues

### Evaluation & Results

- **[Final Results](evaluation/final-results.md)** - Comprehensive evaluation results after Phase 8 improvements
  - Retrieval metrics (Precision, Recall, MRR)
  - Generation quality metrics
  - Performance benchmarks
  - Before/after comparisons

### Phase Documentation

- **[Phase 8: UI Integration](phases/phase-8-ui.md)** - Streamlit web interface implementation details
  - Pipeline visualization
  - Metrics dashboard
  - Example queries and features

### Technical Documentation

- **[System Architecture](technical/architecture.md)** - **📐 COMPREHENSIVE GUIDE**
  - Complete system architecture (all 5 layers)
  - Key strategies & innovations (query expansion, context chaining, etc.)
  - Full tech stack breakdown
  - Detailed folder/file structure
  - Data flow diagrams
  - Design patterns and decisions
  - Performance characteristics

- **[MS Teams Integration Fix](technical/ms-teams-fix.md)** - Critical Pinecone metadata limitation fix
  - 40KB metadata limit workaround
  - Content/metadata mapping pattern
  - Impact on integration queries

- **[TOC Handling](technical/toc-handling.md)** - Table of Contents filtering strategy
  - TOC detection and marking
  - Filtering during retrieval

## 🚀 Getting Started (Quick Path)

### First Time Setup

1. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API keys** - See [API Keys Guide](setup/api-keys.md)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Validate configuration**
   ```bash
   python -m config.settings
   ```

4. **Launch the UI**
   ```bash
   ./run_app.sh
   # Or: streamlit run app.py
   ```

5. **Try example queries** - See [Quick Start UI](guides/quick-start-ui.md)

### Running Tests

See [Getting Started Guide](setup/getting-started.md) for comprehensive testing instructions.

## 📊 System Overview

- **Purpose**: Maximum-quality RAG system for complex multi-topic queries across 2,300+ pages of Watermelon documentation
- **Key Features**: Query decomposition + hierarchical chunking + multi-step retrieval + advanced generation
- **Status**: Phases 1-8 complete (89% overall), Phase 9 (Deployment) pending
- **Performance**:
  - Precision@10: 0.667
  - Recall@10: 0.638
  - MRR: 0.854
  - Avg query time: 27.7s

See [Final Evaluation Results](evaluation/final-results.md) for complete metrics.

## 🔧 Troubleshooting

### Common Issues

1. **Module import errors** - Use `python -m src.module.name` not `python src/module/name.py`
2. **Settings errors** - Run `python -m config.settings` to validate
3. **Empty retrieval results** - See [MS Teams Fix](technical/ms-teams-fix.md) for content mapping
4. **Rate limits** - Groq free tier: ~14 queries/day (See [Quality Guide](guides/quality-improvement.md))

### Quality Issues

See [Quality Improvement Guide](guides/quality-improvement.md) for systematic diagnosis.

## 📖 Root Documentation Files

Files in the project root:

- **[README.md](../README.md)** - Project overview and quick start
- **[CLAUDE.md](../CLAUDE.md)** - Guidance for Claude Code AI assistant (1,300+ lines)

## 🗂️ Other Important Files

- **[requirements.txt](../requirements.txt)** - Python dependencies
- **[.env.example](../.env.example)** - Environment variable template
- **[run_app.sh](../run_app.sh)** - Quick launcher for Streamlit UI

## 🔗 Related Resources

- **Test Queries**: `tests/test_queries.json` - 30 complex test queries
- **Evaluation Results**: `tests/results/comprehensive_evaluation.json` - Raw evaluation data
- **Configuration**: `config/settings.py` - All system settings

---

**Last Updated**: November 4, 2025
**Documentation Version**: 1.0
**System Version**: Phase 8 Complete (89%)
