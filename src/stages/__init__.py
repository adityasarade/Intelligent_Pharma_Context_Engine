"""Pipeline stages for the Pharma Context Engine."""

from .preprocessing import preprocess_pipeline
from .ocr import run_ocr, run_ocr_with_data, extract_text_blocks
from .barcode import decode_all, decode_barcodes, decode_datamatrix, parse_gs1_barcode, validate_ocr_with_barcode
from .verification import KnowledgeBase, OpenFDAClient, RxNormClient
from .enrichment import EnrichmentEngine

__all__ = [
    # Preprocessing
    "preprocess_pipeline",
    # OCR
    "run_ocr",
    "run_ocr_with_data",
    "extract_text_blocks",
    # Barcode
    "decode_all",
    "decode_barcodes",
    "decode_datamatrix",
    "parse_gs1_barcode",
    "validate_ocr_with_barcode",
    # Verification
    "KnowledgeBase",
    "OpenFDAClient",
    "RxNormClient",
    # Enrichment
    "EnrichmentEngine",
]
