import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Ensure src is in path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_pipeline
from src.ocr import run_ocr
from src.enrichment import EnrichmentEngine
from src.schema import PharmaContextOutput

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        """
        path = Path(image_path)
        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        logger.info(f"--- Processing {path.name} ---")
        
        # 1. Preprocessing
        try:
            logger.info("Stage 1: Preprocessing...")
            stages = preprocess_pipeline(str(path))
            enhanced_image = stages['enhanced_gray']
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None
        
        # 2. OCR
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
            
        # 3. Enrichment
        try:
            logger.info("Stage 3: Enrichment & Verification...")
            result = self.enrichment_engine.enrich_text(clean_text)
            logger.info(f"Enrichment Complete. Confidence: {result.verification.confidence_score}")
            return result
        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            return None

if __name__ == "__main__":
    # Quick test if run directly
    if len(sys.argv) < 2:
        print("Usage: python src/main.py <image_path>")
        sys.exit(1)
        
    image_file = sys.argv[1]
    
    # Load dotenv here for standalone run
    from dotenv import load_dotenv
    load_dotenv()
    
    pipeline = PharmaContextPipeline()
    result = pipeline.process_image(image_file)
    
    if result:
        print("\n=== Final Output ===")
        print(result.model_dump_json(indent=2))
    else:
        print("Pipeline failed.")
