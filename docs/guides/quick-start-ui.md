# Quick Start: Using the Streamlit UI

## 🚀 Launch the Application

### Option 1: Quick Launch (Recommended)
```bash
cd /home/hardik121/wm_help_assistant_2
./run_app.sh
```

### Option 2: Manual Launch
```bash
cd /home/hardik121/wm_help_assistant_2
source venv/bin/activate
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## 💡 Using the Application

### 1. First Time Setup (Automatic)
- App loads and initializes the RAG pipeline (~10-15 seconds)
- You'll see "✅ Pipeline initialized successfully!"
- Sidebar shows system status and metrics

### 2. Ask Your First Question

**Two Ways to Ask**:

**A. Select an Example Query**:
1. Use the dropdown: "Choose an example query or write your own:"
2. Select from 5 pre-loaded examples
3. Query appears in the text area below
4. Click "🚀 Ask Question"

**B. Type Your Own Query**:
1. Type or paste your question in the text area
2. Click "🚀 Ask Question"

### 3. Watch the Magic Happen ✨

You'll see a 4-stage progress indicator:
```
📋 Stage 1/4: Understanding query...
🔍 Stage 2/4: Retrieving relevant content...
✨ Stage 3/4: Generating answer...
✅ Stage 4/4: Validating response...
```

Processing takes ~27 seconds on average.

### 4. View Your Results

The interface shows **3 tabs**:

#### Tab 1: 📝 Answer
- Comprehensive answer to your query
- Relevant images in a grid (up to 9 images)
- Clean, formatted markdown

#### Tab 2: 🔧 Pipeline Details
Four expandable sections showing:
- **Query Understanding**: Type, complexity, sub-questions
- **Multi-Step Retrieval**: Chunks retrieved, sections, timing
- **Answer Generation**: Word count, tokens, citations, confidence
- **Validation**: Quality scores, issues, warnings

#### Tab 3: 📊 Metrics
- Total time and breakdown by stage
- Time distribution with progress bars
- Retrieval quality (chunks, sections)
- Answer quality (word count, citations, images)
- Validation scores (overall, completeness, formatting)

---

## 🎯 Example Queries to Try

### Simple Queries
```
How do I create a no-code block on Watermelon?
What is Autonomous Functional Testing?
```

### Complex Queries
```
How do I create a no-code block and process it for Autonomous Functional Testing?
What are the steps to integrate Shopify with Watermelon?
```

### Integration Queries
```
How do I set up MS Teams integration?
How do I configure live chat handover to a human agent?
```

### Technical Queries
```
How do I set up API testing with parameterization?
What security features does Watermelon offer?
```

---

## ⚙️ Sidebar Features

### System Status
- ✅ Pipeline Ready
- 💾 Indexed: 2,106 chunks
- 📄 Documentation: 2,257 pages

### Performance Metrics
Shows latest evaluation results:
- Precision@10: **0.667** (+19%)
- Recall@10: **0.638** (+43%)
- MRR: **0.854** (+49%)
- Avg Time: **27.7s**
- Quality: **100% Excellent**

### Active Improvements
- ✅ Query Expansion (3x variations)
- ✅ Fine-tuned Decomposition

### Configuration
Toggle visibility of:
- Pipeline Stages
- Citations
- Performance Metrics

---

## 🔄 Tips & Tricks

### Getting Better Results
1. **Be Specific**: "How do I set up MS Teams integration?" vs "Tell me about integrations"
2. **Ask Multi-Part Questions**: The system excels at complex, multi-topic queries
3. **Use Full Feature Names**: "Autonomous Functional Testing" vs "AFT"

### Understanding the Pipeline

**Query Understanding**:
- Complex queries are broken into 2-4 sub-questions
- Each sub-question is processed independently
- Results are combined for comprehensive answers

**Query Expansion**:
- Your query is expanded into 3 variations
- Each variation uses synonyms (e.g., "integrate" → "connect")
- This improves recall by 42.8%!

**Hybrid Search**:
- Combines Vector search (semantic) + BM25 (keyword)
- Each finds 30 results
- Reciprocal Rank Fusion combines them
- Cohere reranks to top 10 most relevant

---

## 📊 Interpreting Metrics

### Retrieval Metrics

**Total Retrieved**: Number of chunks found across all sub-questions
- Good: 15-30 chunks
- Too few: Query might be too specific
- Too many: Query might be too broad

**Final Chunks**: Deduplicated chunks in final context
- Ideal: 10-20 chunks
- Ensures comprehensive but focused context

**Unique Sections**: Different documentation sections covered
- Higher = more comprehensive answer
- Shows breadth of retrieval

### Generation Metrics

**Word Count**: Answer length
- Ideal: 400-600 words
- <300: Might be incomplete
- >800: Might be too verbose

**Citations**: Number of source references
- More citations = more verifiable answer
- Shows which chunks were most useful

**Tokens Used**: LLM tokens consumed
- Informational only
- Groq provides free inference

### Validation Scores

**Overall Score**: Combined quality (0-1)
- 0.85+: Excellent ✅
- 0.70-0.85: Good
- <0.70: Needs improvement

**Completeness**: How well sub-questions answered (0-1)
- 1.0 = Perfect
- 0.8+ = Very good
- <0.7 = Some gaps

**Formatting**: Structure and readability (0-1)
- 0.9+ = Well-formatted
- 0.7-0.9 = Acceptable
- <0.7 = Formatting issues

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Make sure you're in the right directory
cd /home/hardik121/wm_help_assistant_2

# Activate virtual environment
source venv/bin/activate

# Try launching again
streamlit run app.py
```

### "Connection Error" or "API Error"
- Check your internet connection
- Verify API keys in `.env` file are correct
- Check if you've hit rate limits (Groq: ~15 queries/day on free tier)

### Query Takes Too Long
- First query: 30-60 seconds (initializing)
- Subsequent queries: 25-30 seconds (normal)
- If much slower: Check internet connection

### Empty Answer
- Query might be too vague or broad
- Try being more specific
- Use full feature/integration names

---

## 🎉 You're Ready!

Just run:
```bash
./run_app.sh
```

And start asking questions about Watermelon documentation!

---

**Need More Help?**
- Full documentation: `PHASE_8_COMPLETE.md`
- Development guide: `CLAUDE.md`
- General info: `README.md`

**Happy Querying! 🍉**
