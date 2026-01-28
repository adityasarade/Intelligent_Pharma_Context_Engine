# Intelligent Pharma-Context Engine

An end-to-end prototype for ingesting pharmaceutical packaging photos, extracting text via OCR, and enriching it with authoritative medical data (OpenFDA, RxNorm) and LLM-generated clinical summaries (Gemini).

## Features
- **Stage 1 (Detection)**: Robust image preprocessing (CLAHE, Bilateral Filter) and Tesseract OCR.
- **Stage 2 (Verification)**: Cross-references drug names with RxNorm and OpenFDA APIs.
- **Stage 3 (Enrichment)**: Uses Google Gemini (2.0 Flash) to extract structured entities and generate clinical context.
- **Resilience**: Handles API rate limits and OCR errors gracefully.

## Setup

### Prerequisites
- Python 3.9+
- Tesseract OCR installed and in PATH (or configured in `.env`).

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # OR using uv
   uv pip install -r pyproject.toml
   ```
   *Note: If you encounter `ModuleNotFoundError: No module named 'cv2'`, run `pip install opencv-python-headless`.*

3. Set up environment variables:
   Create a `.env` file in the root directory:
   ```ini
   GEMINI_API_KEY=your_api_key_here
   # TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe (Optional if in PATH)
   ```

## Usage

### CLI
Process a single image:
```bash
python cli.py process path/to/image.jpg --output result.json
```

### Library
```python
from src.main import PharmaContextPipeline

pipeline = PharmaContextPipeline()
result = pipeline.process_image("data/test/sample.jpg")
print(result.model_dump_json(indent=2))
```

## Structure
- `src/preprocessing.py`: Image enhancement pipeline.
- `src/ocr.py`: Tesseract wrapper.
- `src/knowledge.py`: Clients for OpenFDA and RxNorm.
- `src/enrichment.py`: Gemini LLM logic.
- `src/main.py`: Main orchestration.
- `src/eval.py`: Evaluation metrics (CER, EMR).

## Evaluation
Run the evaluation script to see metrics usage:
```bash
python src/eval.py
```
See `docs/performance_report.md` for detailed results.

## Status
- **Day 1**: ✅ OCR Verified.
- **Day 2**: ✅ Enrichment & Knowledge Base Verified.
- **Day 3**: ✅ CLI & Pipeline Integration Complete.
