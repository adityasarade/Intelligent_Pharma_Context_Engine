# Intelligent Pharma-Context Engine

An end-to-end prototype that ingests photos of pharmaceutical packaging (bottles, blister strips), extracts structured metadata, verifies against authoritative sources (openFDA, RxNorm), and enriches the record with clinical data.

**STATUS**: Pre-Alpha / Planning Phase

## Setup

1.  **Prerequisites**: Python 3.9+, `uv` (recommended).
2.  **Installation**:
    ```bash
    uv sync
    ```
3.  **Configuration**:
    - Copy `.env.example` to `.env` and fill in your `GEMINI_API_KEY`.
    - Ensure Tesseract OCR is installed on your system and accessible via PATH.

## Project Structure

- `src/`: Source code for the pipeline.
- `data/`: Dataset storage (excluded from git).
- `notebooks/`: Exploratory analysis.
- `docs/`: Documentation and periodic reports.

## Datasets

This project uses the following public datasets (please download and place in `data/raw/`):
- [Medicine Bottle Dataset](https://universe.roboflow.com/project-ko6pf/medicine-bottle)
- [Pills Inside Bottles](https://huggingface.co/datasets/gwenxin/pills_inside_bottles)

## License
MIT
