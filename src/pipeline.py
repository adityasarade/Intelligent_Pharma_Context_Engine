"""
Main Pipeline Orchestrator for the Pharma Context Engine.

Coordinates the three stages:
1. Detection & Extraction (preprocessing, OCR, barcode)
2. Verification (knowledge base lookup)
3. Enrichment (LLM clinical context)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from uuid import uuid4

from .stages.preprocessing import preprocess_pipeline
from .stages.ocr import run_ocr, extract_text_blocks
from .stages.barcode import decode_all, validate_ocr_with_barcode, parse_gs1_barcode
from .stages.enrichment import EnrichmentEngine
from .models.schema import (
    PharmaContextOutput,
    Stage1Output,
    OCRChunk,
    BarcodeResult,
    Metrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PharmaContext")


class PharmaContextPipeline:
    """
    Orchestrates the End-to-End Pharma Context Engine.

    Stage 1: Image Preprocessing + OCR + Barcode Detection
    Stage 2: Verification against OpenFDA/RxNorm
    Stage 3: Clinical Enrichment via LLM
    """

    def __init__(self):
        try:
            self.enrichment_engine = EnrichmentEngine()
            logger.info("Enrichment Engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Enrichment Engine: {e}")
            raise

    def process_image(
        self, image_path: str, ground_truth: Optional[Dict[str, Any]] = None
    ) -> Optional[PharmaContextOutput]:
        """
        Runs the full pipeline on a single image.

        Args:
            image_path: Path to the input image file
            ground_truth: Optional ground truth for computing metrics

        Returns:
            PharmaContextOutput with all extracted and enriched data,
            or None if processing failed
        """
        start_time = time.time()
        path = Path(image_path)

        if not path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        image_id = f"IMG_{uuid4().hex[:8].upper()}"
        logger.info(f"--- Processing {path.name} (ID: {image_id}) ---")

        # =====================
        # STAGE 1: Detection & Extraction
        # =====================
        stage1 = Stage1Output()

        # 1a. Preprocessing
        try:
            logger.info("Stage 1a: Preprocessing...")
            stages = preprocess_pipeline(str(path))
            enhanced_image = stages["enhanced_gray"]
            original_image = stages["original"]
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None

        # 1b. OCR
        try:
            logger.info("Stage 1b: OCR...")
            raw_text = run_ocr(enhanced_image)
            stage1.raw_text = raw_text.strip()

            # Get detailed OCR chunks with bounding boxes
            text_blocks = extract_text_blocks(enhanced_image)
            for block in text_blocks:
                stage1.ocr_chunks.append(
                    OCRChunk(
                        text=block["text"],
                        bbox=block["bbox"],
                        confidence=float(block["conf"]),
                        block_num=block.get("block_num"),
                        line_num=block.get("line_num"),
                    )
                )

            if not stage1.raw_text:
                logger.warning("OCR extracted empty text.")
            else:
                logger.info(f"OCR extracted {len(stage1.raw_text)} characters.")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return None

        # 1c. Barcode Detection
        try:
            logger.info("Stage 1c: Barcode Detection...")
            barcodes = decode_all(original_image)
            for bc in barcodes:
                parsed = parse_gs1_barcode(bc["data"]) if bc["data"] else None
                stage1.barcodes.append(
                    BarcodeResult(
                        type=bc["type"],
                        data=bc["data"],
                        bbox=bc.get("bbox"),
                        decode_confidence=bc.get("decode_confidence", 0.95),
                        parsed=parsed,
                    )
                )
            logger.info(f"Found {len(stage1.barcodes)} barcode(s).")
        except Exception as e:
            logger.warning(f"Barcode detection failed (non-critical): {e}")

        # =====================
        # STAGE 2: Entity Extraction & Verification
        # =====================
        try:
            logger.info("Stage 2: Entity Extraction & Verification...")

            # Extract entities using LLM
            extracted_entities = self.enrichment_engine.extract_entities(
                stage1.raw_text
            )

            # Validate with barcode if available
            barcode_validation = None
            if stage1.barcodes:
                barcode_data = [
                    {"data": bc.data, "type": bc.type} for bc in stage1.barcodes
                ]
                barcode_validation = validate_ocr_with_barcode(
                    extracted_entities.model_dump(), barcode_data
                )

            # Verify against knowledge bases
            verification = self.enrichment_engine.verify_entities(
                extracted_entities, barcode_validation
            )

            logger.info(
                f"Verification Complete. Confidence: {verification.confidence_score:.2f}"
            )
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return None

        # =====================
        # STAGE 3: Clinical Enrichment
        # =====================
        try:
            logger.info("Stage 3: Clinical Enrichment...")

            # Determine the best drug name to use for enrichment
            final_drug_name = (
                verification.openfda.brand_name
                or verification.rxnorm.name
                or extracted_entities.drug_name
                or extracted_entities.generic_name
            )

            if final_drug_name:
                clinical_context = self.enrichment_engine.generate_clinical_context(
                    final_drug_name
                )
                # Generate human-readable summary
                clinical_context.human_summary = (
                    self.enrichment_engine.generate_human_summary(
                        final_drug_name, clinical_context
                    )
                )
            else:
                from .models.schema import ClinicalContext
                clinical_context = ClinicalContext()
                logger.warning("No drug name found for clinical enrichment.")

        except Exception as e:
            logger.error(f"Enrichment failed: {e}")
            from .models.schema import ClinicalContext
            clinical_context = ClinicalContext()

        # =====================
        # Build Final Output
        # =====================
        processing_time = (time.time() - start_time) * 1000

        provenance = self.enrichment_engine.create_provenance(
            extracted_entities.drug_name or extracted_entities.generic_name,
            final_drug_name if "final_drug_name" in dir() else None,
        )

        output = PharmaContextOutput(
            image_id=image_id,
            source_image_path=str(path.absolute()),
            capture_datetime=datetime.utcnow(),
            stage1=stage1,
            extracted_entities=extracted_entities,
            verification=verification,
            clinical_enrichment=clinical_context,
            provenance=provenance,
            metrics=Metrics(processing_time_ms=processing_time),
        )

        logger.info(f"Pipeline complete in {processing_time:.0f}ms")
        return output


if __name__ == "__main__":
    import sys

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
