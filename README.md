# Intelligent Pharma-Context Engine

An end-to-end pipeline for extracting metadata from pharmaceutical packaging images, verifying against authoritative databases, and enriching with clinical information.

## Overview

This prototype addresses the challenge of digital pharmacy safety by:
1. **Extracting** text and barcodes from medicine bottles/strips
2. **Verifying** against authoritative sources (OpenFDA, RxNorm)
3. **Enriching** with clinical metadata (storage, warnings, side effects)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Raw Image                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                    STAGE 1: Detection & Extraction                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Preprocessing│──│  Tesseract   │──│    Barcode Decoder       │  │
│  │ CLAHE/Filter │  │     OCR      │  │   (pyzbar/pylibdmtx)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                    STAGE 2: Verification                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Gemini LLM   │──│   OpenFDA    │──│       RxNorm             │  │
│  │ Entity Extr. │  │   Lookup     │  │       Lookup             │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│                          │                                          │
│               ┌──────────▼──────────┐                              │
│               │   Fuzzy Matching    │ (RapidFuzz for OCR errors)   │
│               │   Barcode Validation│                              │
│               └─────────────────────┘                              │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                    STAGE 3: Enrichment                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Gemini LLM Clinical Context                      │  │
│  │  • Indications    • Contraindications    • Storage            │  │
│  │  • Warnings       • Side Effects         • Human Summary      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                    OUTPUT: Enriched JSON                            │
│  { image_id, stage1, extracted_entities, verification,             │
│    clinical_enrichment, provenance, metrics }                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Image Preprocessing
- **CLAHE**: Handles uneven lighting and low contrast
- **Bilateral Filtering**: Reduces noise while preserving text edges
- **Shadow Removal**: Morphological operations for glare/shadow correction

### OCR Strategy
- **Tesseract OCR** with PSM 6 (single uniform block) for label text
- Returns bounding boxes for spatial analysis
- Confidence scores for quality filtering

### Handling OCR Errors (Fuzzy Entity Resolution)
- **RapidFuzz** for edit-distance matching (handles "Lisinopri1" → "Lisinopril")
- Combined confidence scoring from multiple signals
- Human review flag for low-confidence matches

### Layout Agnosticism
- LLM-based entity extraction (Gemini) understands context without hard-coded coordinates
- No template matching - works across diverse packaging formats

### Multi-Modal Validation
- Barcode decoding (pyzbar for 1D/2D, pylibdmtx for DataMatrix)
- GS1/NDC parsing for pharmaceutical identifiers
- Barcode data boosts verification confidence when available

## Project Structure

```
├── src/
│   ├── pipeline.py              # Main orchestration
│   ├── stages/
│   │   ├── preprocessing.py     # Image enhancement
│   │   ├── ocr.py               # Tesseract OCR
│   │   ├── barcode.py           # Barcode/DataMatrix decoding
│   │   ├── verification.py      # OpenFDA/RxNorm clients
│   │   └── enrichment.py        # Gemini LLM integration
│   ├── models/
│   │   └── schema.py            # Pydantic output schemas
│   └── evaluation/
│       └── metrics.py           # CER/EMR calculations
├── tests/                       # Unit & integration tests
├── notebooks/
│   └── demo_pipeline.ipynb      # Full demo notebook
├── data/
│   └── raw/                     # Medicine bottle datasets
├── docs/
│   └── performance_report.md    # Evaluation results
└── docker/                      # Reproducible environment
```

## Installation

### Prerequisites
- Python 3.9+
- Tesseract OCR ([install guide](https://github.com/tesseract-ocr/tesseract))
- Gemini API key

### Setup
```bash
# Clone and install
git clone <repo-url>
cd HealthInfinityAI

# Install dependencies
pip install -e .

# Or with barcode support
pip install -e ".[barcode]"

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Usage

### CLI
```bash
python cli.py process path/to/image.jpg --output result.json
```

### Python API
```python
from src import PharmaContextPipeline

pipeline = PharmaContextPipeline()
result = pipeline.process_image("medicine_bottle.jpg")
print(result.model_dump_json(indent=2))
```

### Jupyter Notebook
See `notebooks/demo_pipeline.ipynb` for a complete walkthrough.

## Output Schema

```json
{
  "image_id": "IMG_ABC12345",
  "source_image_path": "/path/to/image.jpg",
  "stage1": {
    "ocr_chunks": [{"text": "...", "bbox": [x1,y1,x2,y2], "confidence": 85}],
    "raw_text": "AMOXICILLIN 500mg...",
    "barcodes": [{"type": "DataMatrix", "data": "...", "parsed": {...}}]
  },
  "extracted_entities": {
    "drug_name": "Amoxicillin",
    "manufacturer": "Teva",
    "strength": "500 mg",
    "composition": "Amoxicillin Trihydrate"
  },
  "verification": {
    "openfda": {"is_found": true, "brand_name": "AMOXICILLIN", ...},
    "rxnorm": {"is_found": true, "rxcui": "723", ...},
    "confidence_score": 0.85,
    "human_review_needed": false
  },
  "clinical_enrichment": {
    "indications": ["Bacterial infections", ...],
    "warnings": ["Allergic reactions", ...],
    "side_effects": ["Nausea", "Diarrhea", ...],
    "storage": "Store below 25°C"
  },
  "provenance": {...},
  "metrics": {"processing_time_ms": 2340}
}
```

## Evaluation

### Metrics
- **CER (Character Error Rate)**: `(S + D + I) / N` where S=substitutions, D=deletions, I=insertions
- **Entity Match Rate**: Percentage of correctly matched fields against ground truth

### Running Tests
```bash
# Unit tests
pytest -m "not integration"

# All tests including API calls
pytest
```

See `docs/performance_report.md` for detailed evaluation results.

## Data Sources

| Source | Type | License |
|--------|------|---------|
| [Medicine Bottle Dataset](https://universe.roboflow.com/project-ko6pf/medicine-bottle) | Images | CC BY 4.0 |
| [Pills Inside Bottles](https://huggingface.co/datasets/gwenxin/pills_inside_bottles) | Images | CC BY 4.0 |
| [OpenFDA Drug Labels](https://open.fda.gov/apis/drug/label/) | API | Public Domain |
| [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/) | API | Public Domain |

## License

MIT License - see [LICENSE](LICENSE)
