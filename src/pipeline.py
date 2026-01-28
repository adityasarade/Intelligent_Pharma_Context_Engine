"""
Main Pipeline Orchestrator for the Pharma Context Engine.

Coordinates the three stages:
1. Preprocessing (image enhancement)
2. OCR (text extraction)
3. Verification & Enrichment (knowledge base + LLM)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .stages.preprocessing import preprocess_pipeline
from .stages.ocr import run_ocr
from .stages.enrichment import EnrichmentEngine
from .models.schema import PharmaContextOutput


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PharmaContext")


class PharmaContextPipeline:
    """
    Orchestrates the End-to-End Pharma Context Engine.

    Stage 1: Image Preprocessing
    Stage 2: OCR
    Stage 3: Verification & Enrichment
    """

    def __init__(self):
        try:
            self.enrichment_engine = EnrichmentEngine()
            logger.info("Enrichment Engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Enrichment Engine: {e}")
            raise

    def process_image(self, image_path: str) -> Optional[PharmaContextOutput]:
        """
        Runs the full pipeline on a single image.

        Args:
            image_path: Path to the input image file

        Returns:
            PharmaContextOutput with all extracted and enriched data,
            or None if processing failed
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        logger.info(f"--- Processing {path.name} ---")

        # Stage 1: Preprocessing
        try:
            logger.info("Stage 1: Preprocessing...")
            stages = preprocess_pipeline(str(path))
            enhanced_image = stages["enhanced_gray"]
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None

        # Stage 2: OCR
        try:
            logger.info("Stage 2: OCR...")
            raw_text = run_ocr(enhanced_image)
            clean_text = raw_text.strip()
            if not clean_text:
                logger.warning("OCR extracted empty text.")
            else:
                logger.info(f"OCR extracted {len(clean_text)} characters.")
                logger.debug(f"Raw Text Snippet: {clean_text[:50]}...")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

        # Stage 3: Enrichment
        try:
            logger.info("Stage 3: Enrichment & Verification...")
            result = self.enrichment_engine.enrich_text(clean_text)
            logger.info(
                f"Enrichment Complete. "
                f"Confidence: {result.verification.confidence_score}"
            )
            return result
        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            return None


if __name__ == "__main__":
    # Quick test if run directly
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipeline <image_path>")
        sys.exit(1)

    image_file = sys.argv[1]

    from dotenv import load_dotenv

    load_dotenv()

    pipeline = PharmaContextPipeline()
    result = pipeline.process_image(image_file)

    if result:
        print("\n=== Final Output ===")
        print(result.model_dump_json(indent=2))
    else:
        print("Pipeline failed.")
