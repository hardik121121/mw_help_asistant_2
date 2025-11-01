# Next Steps: Resume Document Processing

**Current Status**: Environment setup complete, Phase 2 code ready, but PDF processing was paused due to slow performance.

---

## Quick Start (Recommended)

### Option 1: Fast Processing (15-30 min) - Best for Testing

**Step 1**: Disable image extraction for faster processing

Edit `src/ingestion/docling_processor.py` line ~490:

```python
# CHANGE THIS:
processor = DoclingPDFProcessor(
    pdf_path=str(pdf_path),
    image_output_dir=str(image_dir),
    enable_tables=True,
    enable_images=True,     # ← Change to False
    enable_ocr=False        # ← Already False (good)
)

# TO THIS:
processor = DoclingPDFProcessor(
    pdf_path=str(pdf_path),
    image_output_dir=str(image_dir),
    enable_tables=True,
    enable_images=False,    # ✓ Disabled → faster
    enable_ocr=False        # ✓ Already disabled → faster
)
```

**Step 2**: Run processing

```bash
cd /home/hardik121/wm_help_assistant_2
source venv/bin/activate
python -m src.ingestion.docling_processor
```

**Expected**: 15-30 minutes (vs 60-90 min with images)

**Step 3**: After completion, run remaining Phase 2 steps

```bash
# Create hierarchical chunks (~1-2 min)
python -m src.ingestion.hierarchical_chunker

# Evaluate quality (<1 min)
python -m src.ingestion.chunk_evaluator

# View report
cat tests/results/chunk_quality_report.txt
```

**Success**: Overall quality score should be >0.80

---

### Option 2: Full Processing Overnight (60-90 min)

Keep all features enabled (images + OCR) and run in background:

```bash
cd /home/hardik121/wm_help_assistant_2
source venv/bin/activate

# Run in background
nohup python -m src.ingestion.docling_processor > docling_full.log 2>&1 &

# Note the process ID (shown after command)
# Check progress anytime:
tail -f docling_full.log

# Check if still running:
ps aux | grep docling_processor
```

**Note**: You can log out and it will keep running. Check `docling_full.log` when you return.

---

### Option 3: Test with Small Subset (2 min)

Extract first 20 pages to verify pipeline works:

```bash
cd /home/hardik121/wm_help_assistant_2
source venv/bin/activate

# Install PyPDF2
pip install pypdf

# Extract first 20 pages
python -c "
from PyPDF2 import PdfReader, PdfWriter
reader = PdfReader('data/helpdocs.pdf')
writer = PdfWriter()
for i in range(20):
    writer.add_page(reader.pages[i])
with open('data/helpdocs_test.pdf', 'wb') as f:
    writer.write(f)
print('Created data/helpdocs_test.pdf with first 20 pages')
"

# Temporarily update .env to use test PDF
nano .env
# Change: PDF_PATH=data/helpdocs_test.pdf

# Run processing (should take <2 minutes)
python -m src.ingestion.docling_processor

# Don't forget to change .env back to helpdocs.pdf when testing full version
```

---

## After Processing Completes

**You should see**:
- ✅ `cache/docling_processed.json` (23-25 MB)
- ✅ `cache/images/` directory with PNG files (if images enabled)
- ✅ Log showing: "PDF processing complete", "Extracted: X text blocks, Y images, Z tables"

**Run the remaining steps**:

```bash
# Step 2: Hierarchical chunking
python -m src.ingestion.hierarchical_chunker
# Output: cache/hierarchical_chunks.json (~3 MB)

# Step 3: Quality evaluation
python -m src.ingestion.chunk_evaluator
# Output: tests/results/chunk_quality_report.txt

# View the report
cat tests/results/chunk_quality_report.txt
```

**Look for**:
- Overall Quality Score: **>0.80** (target threshold)
- Size consistency: >0.85
- Structure preservation: >0.90
- Context completeness: >0.85

---

## If Quality Score is Too Low (<0.80)

Edit `.env` and adjust chunking parameters:

```bash
# Try increasing chunk size
CHUNK_SIZE=2000  # Was 1500

# Or reduce overlap
CHUNK_OVERLAP=200  # Was 300

# Then re-run chunker
python -m src.ingestion.hierarchical_chunker
python -m src.ingestion.chunk_evaluator
```

---

## Troubleshooting

### Process appears stuck?
```bash
# Check if it's actually running (high CPU = working)
top -p $(pgrep -f docling_processor)

# Or
ps aux | grep docling_processor
```

If CPU is >300%, it's working (just slow). Docling processes silently for large PDFs.

### Out of memory?
Large PDFs need 4-8 GB RAM. If you run out:
- Close other applications
- Or process in smaller batches (Option 3)

### Import errors?
```bash
# Make sure venv is activated
source venv/bin/activate

# Verify Docling is installed
pip list | grep docling
```

---

## After Phase 2 is Complete

**Then proceed to Phase 3**: Query Understanding Engine
- Query decomposition
- Query classification
- Intent understanding

See `PROGRESS.md` for Phase 3-9 details.

---

## Summary

**Recommended path**:
1. Edit `docling_processor.py` to disable images (line ~490)
2. Run: `python -m src.ingestion.docling_processor` (wait 15-30 min)
3. Run: `python -m src.ingestion.hierarchical_chunker` (wait 1-2 min)
4. Run: `python -m src.ingestion.chunk_evaluator` (wait <1 min)
5. Check quality score: `cat tests/results/chunk_quality_report.txt`
6. If score >0.80 → Ready for Phase 3!

**Alternative**: Run overnight with all features enabled using Option 2.

---

**Questions?** Check `CLAUDE.md` for architecture details, `SETUP.md` for setup help, or `PROGRESS.md` for technical details.
