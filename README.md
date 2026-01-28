# Intelligent Pharma-Context Engine

An end-to-end prototype for ingesting pharmaceutical packaging photos, extracting text via OCR, and enriching it with authoritative medical data (OpenFDA, RxNorm) and LLM-generated clinical summaries (Gemini).

## Features

- **Stage 1 (Detection)**: Robust image preprocessing (CLAHE, Bilateral Filter) and Tesseract OCR
- **Stage 2 (Verification)**: Cross-references drug names with RxNorm and OpenFDA APIs
- **Stage 3 (Enrichment)**: Uses Google Gemini to extract structured entities and generate clinical context
- **Resilience**: Handles API rate limits and OCR errors gracefully

## Project Structure

```
├── src/                          # Source code
│   ├── pipeline.py               # Main orchestration
│   ├── stages/                   # Pipeline stages
│   │   ├── preprocessing.py      # Image enhancement
│   │   ├── ocr.py                # Tesseract OCR
│   │   ├── verification.py       # OpenFDA/RxNorm clients
│   │   └── enrichment.py         # Gemini LLM integration
│   ├── models/                   # Pydantic schemas
│   │   └── schema.py             # Output data models
│   └── evaluation/               # Metrics
│       └── metrics.py            # CER/EMR calculations
├── tests/                        # Unit & integration tests
├── data/                         # Datasets
│   ├── raw/                      # Raw image datasets
│   ├── processed/                # Processed outputs
│   └── test/                     # Test images
├── docs/                         # Documentation
├── docker/                       # Docker configuration
├── models/                       # Saved model checkpoints
├── notebooks/                    # Jupyter notebooks
└── cli.py                        # Command-line interface
```

## Setup

### Prerequisites

- Python 3.9+
- Tesseract OCR installed and in PATH (or configured in `.env`)

### Installation

1. Clone the repository

2. Install dependencies:
   ```bash
   # Using uv (recommended)
   uv pip install -e .

   # Or using pip
   pip install -e .
   ```

3. Set up environment variables by creating a `.env` file:
   ```ini
   GEMINI_API_KEY=your_api_key_here
   # TESSERACT_CMD=/path/to/tesseract (Optional if in PATH)
   ```

## Usage

### CLI

Process a single image:
```bash
python cli.py process path/to/image.jpg --output result.json
```

### Library

```python
from src import PharmaContextPipeline

pipeline = PharmaContextPipeline()
result = pipeline.process_image("data/test/sample.jpg")
print(result.model_dump_json(indent=2))
```

### Docker

```bash
cd docker
docker-compose up --build
```

## Testing

Run all tests:
```bash
pytest

# Skip integration tests (no network required)
pytest -m "not integration"

# With coverage
pytest --cov=src
```

## Evaluation Metrics

- **CER (Character Error Rate)**: Measures OCR accuracy
- **EMR (Entity Match Rate)**: Measures field extraction accuracy

See `docs/performance_report.md` for detailed results.

## Data Sources

- **OpenFDA Drug Labels**: Authoritative FDA drug information
- **RxNorm**: NIH normalized drug names and RxCUI codes
- **Medicine Bottle Dataset**: CC BY 4.0 licensed images for training/testing

## License

MIT License - see [LICENSE](LICENSE) for details.
