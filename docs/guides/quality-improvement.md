# Quality Improvement Guide

**Problem**: Bot outputs are not meeting quality expectations.

**Root Cause**: From evaluation results, retrieval quality is low (Precision: 0.567, Recall: 0.551) while generation quality is high (0.916). This means **the bot generates good answers when given good context, but isn't finding the right context**.

---

## Quick Diagnostic Checklist

Run these checks to identify the exact problem:

### ✅ Step 1: Check Retrieved Context Quality

```bash
python scripts/diagnose_quality.py "How do I integrate MS Teams?"
```

**Look for**:
- Are top chunks relevant to the query?
- Is content empty (0 chars)?
- Are chunks from the right section of documentation?
- Is the average score > 0.7?

### ✅ Step 2: Check Chunk Metadata

```bash
python -c "
import json
chunks = json.load(open('cache/hierarchical_chunks_filtered.json'))
chunk = chunks[50]  # Check a random chunk
print('Heading:', chunk['metadata'].get('current_heading'))
print('Path:', chunk['metadata'].get('heading_path'))
print('Topics:', chunk['metadata'].get('topics', 'Not set'))
print('Content type:', chunk['metadata'].get('content_type'))
print('Preview:', chunk['content'][:200])
"
```

**Look for**:
- Is `heading_path` accurate?
- Is `content_type` meaningful?
- Does metadata describe what the chunk is about?

### ✅ Step 3: Check Query Understanding

```bash
python -m src.query.query_understanding
# Enter: "How do I create a no-code block and test it?"
```

**Look for**:
- Are sub-questions covering all parts of the query?
- Is classification correct (multi-topic_procedural)?
- Are there at least 2-3 sub-questions for complex queries?

---

## Solution Paths (Choose Based on Diagnosis)

### 🎯 Solution 1: Improve Chunk Metadata (RECOMMENDED FIRST)

**When to use**: If chunks have poor metadata (missing topics, vague headings)

**Impact**: High (20-30% improvement in retrieval precision)

**Effort**: Low (30 minutes)

**Steps**:

1. **Enrich chunks with better metadata**:
   ```bash
   python scripts/enrich_chunks.py
   ```

2. **Review enriched chunks**:
   ```bash
   # Check first 5 chunks
   python -c "
   import json
   chunks = json.load(open('cache/hierarchical_chunks_enriched.json'))
   for i in range(5):
       c = chunks[i]
       print(f'\\nChunk {i}:')
       print(f'  Type: {c[\"metadata\"].get(\"content_type\")}')
       print(f'  Topics: {c[\"metadata\"].get(\"topics\", [])}')
       print(f'  Heading: {c[\"metadata\"].get(\"current_heading\")}')
   "
   ```

3. **If satisfied, replace original chunks**:
   ```bash
   cp cache/hierarchical_chunks_filtered.json cache/hierarchical_chunks_filtered.json.backup
   cp cache/hierarchical_chunks_enriched.json cache/hierarchical_chunks_filtered.json
   ```

4. **Re-index** (embeddings + Pinecone + BM25):
   ```bash
   python -m src.database.run_phase5
   ```

5. **Test improvement**:
   ```bash
   python scripts/diagnose_quality.py "How do I integrate MS Teams?"
   ```

---

### 🎯 Solution 2: Increase Retrieval Coverage

**When to use**: If retrieval is getting < 15 chunks or missing relevant sections

**Impact**: Medium (10-15% improvement in recall)

**Effort**: Very Low (2 minutes)

**Steps**:

1. **Edit `config/settings.py`**:
   ```python
   # Increase from 30 → 50
   vector_top_k: int = 50  # Was 30
   bm25_top_k: int = 50    # Was 30

   # Increase final context
   # Add this in generation config section:
   final_context_chunks: int = 30  # Was 20 (if this setting exists)
   ```

2. **Restart and test**:
   ```bash
   python scripts/diagnose_quality.py "Your test query"
   ```

3. **Check if retrieval count increased** (should see ~30 chunks now instead of 20)

---

### 🎯 Solution 3: Improve Query Decomposition

**When to use**: If complex queries generate only 1 sub-question

**Impact**: Medium (15-20% improvement for complex queries)

**Effort**: Medium (1 hour)

**Steps**:

1. **Edit `src/query/query_decomposer.py`**

2. **Find the decomposition prompt** (search for "Break down the following query")

3. **Improve the prompt**:
   ```python
   # Current prompt is likely too simple. Make it more specific:

   prompt = f"""You are a query analysis expert. Break down the following user query into 2-4 specific sub-questions that would help retrieve comprehensive information.

   **Guidelines**:
   - If the query mentions multiple topics/features, create separate sub-questions for each
   - If the query asks "how to", break it into: (1) what is it? (2) prerequisites, (3) step-by-step guide
   - If the query mentions integrations, ask about setup AND configuration separately
   - Each sub-question should be self-contained and answerable independently

   **User Query**: {query}

   **Examples**:
   Query: "How do I create a no-code block and test it?"
   Sub-questions:
   1. What are no-code blocks in Watermelon?
   2. What are the steps to create a no-code block?
   3. How do I test a no-code block?

   Query: "How do I integrate MS Teams with Watermelon?"
   Sub-questions:
   1. What are the prerequisites for MS Teams integration?
   2. What are the step-by-step setup instructions for MS Teams integration?
   3. How do I configure automated responses in MS Teams integration?

   Now break down the user query above into 2-4 sub-questions:
   """
   ```

4. **Test**:
   ```bash
   python -m src.query.query_decomposer
   # Try several complex queries
   ```

---

### 🎯 Solution 4: Add Query Expansion

**When to use**: If queries use different terminology than documentation

**Impact**: High (20-25% improvement in recall)

**Effort**: Medium (2 hours)

**Steps**:

1. **Create query expansion module**:

```python
# src/query/query_expander.py
from typing import List
import logging

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands queries with synonyms and related terms."""

    def __init__(self):
        # Domain-specific synonym mappings
        self.synonyms = {
            'integrate': ['connect', 'link', 'setup', 'configure'],
            'create': ['make', 'build', 'add', 'set up'],
            'test': ['verify', 'check', 'validate', 'try'],
            'error': ['issue', 'problem', 'bug', 'failure'],
            'setup': ['configure', 'install', 'initialize'],
            'automation': ['workflow', 'trigger', 'automated process'],
            'chatbot': ['bot', 'conversation', 'chat'],
        }

        # Integration name variations
        self.integration_aliases = {
            'MS Teams': ['Microsoft Teams', 'Teams'],
            'Slack': ['Slack messenger'],
            'WhatsApp': ['WhatsApp Business', 'WA'],
        }

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms.

        Returns: List of query variations (original + expanded)
        """
        variations = [query]  # Always include original

        query_lower = query.lower()

        # Add synonym variations
        for term, syns in self.synonyms.items():
            if term in query_lower:
                for syn in syns[:2]:  # Max 2 synonyms per term
                    expanded = query.lower().replace(term, syn)
                    variations.append(expanded)

        # Add integration name variations
        for canonical, aliases in self.integration_aliases.items():
            if canonical.lower() in query_lower:
                for alias in aliases:
                    variations.append(query.replace(canonical, alias))

        # Limit to top 3 variations to avoid noise
        return variations[:3]


# Example usage in hybrid_search.py:
# from src.query.query_expander import QueryExpander
#
# expander = QueryExpander()
# query_variations = expander.expand_query(query)
#
# # Search with all variations and combine results
# all_results = []
# for var in query_variations:
#     results = self.search(var)
#     all_results.extend(results)
#
# # Deduplicate and rerank
# final_results = self.deduplicate_and_rerank(all_results)
```

2. **Integrate into `src/retrieval/hybrid_search.py`**

3. **Test**:
   ```bash
   python -m src.retrieval.hybrid_search
   ```

---

### 🎯 Solution 5: Add Manual Topic Annotations (LAST RESORT)

**When to use**: ONLY if automated solutions don't work

**Impact**: Very High (30-40% improvement)

**Effort**: VERY High (days of manual work)

**Why it's last resort**: Time-consuming, not scalable

**If you must do this**:

1. **Don't annotate ALL pages** - annotate representative samples per topic:
   - 5-10 pages per major topic (integrations, features, setup, etc.)
   - Focus on most frequently asked topics

2. **Use a structured format**:

```json
{
  "page": 45,
  "topic": "MS Teams Integration",
  "subtopics": ["Setup", "Configuration", "Troubleshooting"],
  "key_points": [
    "Requires admin permissions",
    "Uses OAuth authentication",
    "Supports automated responses"
  ],
  "related_pages": [46, 47, 48]
}
```

3. **Inject annotations into chunk metadata**:

```python
# scripts/inject_annotations.py
import json

annotations = json.load(open('manual_annotations.json'))
chunks = json.load(open('cache/hierarchical_chunks_filtered.json'))

for chunk in chunks:
    page = chunk['metadata']['page_start']
    if page in [a['page'] for a in annotations]:
        annotation = next(a for a in annotations if a['page'] == page)
        chunk['metadata']['manual_topic'] = annotation['topic']
        chunk['metadata']['manual_subtopics'] = annotation['subtopics']
        chunk['metadata']['key_points'] = annotation['key_points']

json.dump(chunks, open('cache/hierarchical_chunks_annotated.json', 'w'), indent=2)
```

**But seriously**: Try automated solutions first! Manual annotation should be absolute last resort.

---

## Recommended Improvement Sequence

**Week 1** (Quick wins):
1. ✅ Run diagnostic script on 10 test queries
2. ✅ Enrich chunk metadata (Solution 1)
3. ✅ Increase retrieval coverage (Solution 2)
4. ✅ Test and measure improvement

**Expected**: Precision 0.567 → 0.70+, Recall 0.551 → 0.65+

**Week 2** (If needed):
5. ✅ Improve query decomposition (Solution 3)
6. ✅ Add query expansion (Solution 4)
7. ✅ Test and measure improvement

**Expected**: Precision 0.70 → 0.80+, Recall 0.65 → 0.75+

**Only if desperate** (Week 3+):
8. ⚠️ Manual annotations (Solution 5) - 10-20 key pages only

---

## Measuring Improvement

After each change, run:

```bash
# Quick test (5 queries)
python -m src.evaluation.comprehensive_evaluation
# Enter: 5

# Compare results
python scripts/compare_evaluations.py \
    tests/results/comprehensive_evaluation_before.json \
    tests/results/comprehensive_evaluation.json
```

**Success criteria**:
- Precision@10 > 0.70 (from 0.567)
- Recall@10 > 0.65 (from 0.551)
- MRR > 0.70 (from 0.627)
- Generation quality maintained > 0.90

---

## Common Mistakes to Avoid

❌ **Don't**: Manually annotate everything before trying automated solutions
✅ **Do**: Start with metadata enrichment and retrieval tuning

❌ **Don't**: Change multiple things at once
✅ **Do**: Change one thing, measure, then iterate

❌ **Don't**: Focus on generation quality (it's already excellent at 0.916)
✅ **Do**: Focus on retrieval quality (it's the bottleneck)

❌ **Don't**: Assume you need to rebuild from scratch
✅ **Do**: Incrementally improve what exists

---

## Quick Reference: What to Fix When

**Symptoms → Solutions**:

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Bot gives generic answers | Poor retrieval | Solution 1 + 2 |
| Bot says "I don't know" | No relevant chunks found | Solution 2 + 4 |
| Bot misses parts of question | Poor decomposition | Solution 3 |
| Bot gets wrong topic | Poor metadata | Solution 1 |
| Bot uses wrong terminology | Query-doc mismatch | Solution 4 |
| Everything else failed | Deep structural issues | Solution 5 (last resort) |

---

## Expected Timeline

- **Solution 1**: 30 minutes
- **Solution 2**: 5 minutes
- **Solution 3**: 1-2 hours
- **Solution 4**: 2-3 hours
- **Solution 5**: Days to weeks (avoid if possible)

**Total realistic timeline for major improvement**: 1-2 weeks

---

## Need Help?

1. Run diagnostic: `python scripts/diagnose_quality.py "your query"`
2. Share the output
3. Review this guide's "Symptoms → Solutions" table
4. Start with Solution 1 (metadata enrichment)

**Remember**: Your generation is already excellent (0.916). The problem is retrieval (0.567). Fix retrieval = fix everything.
