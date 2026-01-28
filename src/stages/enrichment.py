"""
Stage 3: Enrichment

Uses Gemini LLM to:
1. Extract structured entities from OCR text
2. Generate clinical context (indications, warnings, side effects)
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from ..models.schema import (
    PharmaContextOutput,
    EntityExtraction,
    VerificationData,
    ClinicalContext,
    OpenFDAResult,
    RxNormResult,
)
from .verification import KnowledgeBase


load_dotenv()
logger = logging.getLogger(__name__)


class EnrichmentEngine:
    """
    Orchestrates entity extraction and clinical enrichment using Gemini LLM.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is missing.")

        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-2.0-flash"
        self.model = genai.GenerativeModel(self.model_name)
        self.kb = KnowledgeBase()

    def _call_llm_with_retry(
        self, prompt: str, max_retries: int = 3
    ) -> Optional[str]:
        """Helper to call LLM with retry logic for rate limits (429)."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (attempt + 1) * 5
                    logger.warning(
                        f"Rate limit hit. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed: {e}")
                    return None
        logger.error("Max retries exceeded for LLM call.")
        return None

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"LLM JSON Parse Error. Content: {content}")
            return None

    def _extract_entities_with_llm(self, text: str) -> EntityExtraction:
        """
        Uses Gemini to extract entities from raw OCR text.
        """
        prompt = f"""
        You are an expert pharmaceutical entity extractor.
        Extract the following fields from the provided OCR text of a medicine bottle:
        - brand_name
        - generic_name
        - strength
        - dosage_form
        - manufacturer

        Return the result as a valid JSON object matching this structure:
        {{
            "brand_name": "...",
            "generic_name": "...",
            "strength": "...",
            "dosage_form": "...",
            "manufacturer": "..."
        }}

        If a field is not found, set it to null. Do not hallucinate.

        OCR Text:
        {text}
        """

        content = self._call_llm_with_retry(prompt)
        if not content:
            return EntityExtraction()

        data = self._parse_json_response(content)
        if data:
            return EntityExtraction(**data)
        return EntityExtraction()

    def _generate_clinical_context(
        self, drug_name: str, verified_info: Dict
    ) -> ClinicalContext:
        """
        Generates clinical context using Gemini, grounded by verified drug name.
        """
        prompt = f"""
        Provide a brief, professional clinical summary for the drug: "{drug_name}".

        Include:
        1. Indications (List of main uses)
        2. Contraindications (List of when NOT to use)
        3. Warnings (Major precautions)
        4. Common Side Effects

        Format as JSON:
        {{
            "indications": ["..."],
            "contraindications": ["..."],
            "warnings": ["..."],
            "side_effects": ["..."]
        }}

        Strictly adhere to medical facts. If unsure, return empty lists.
        """

        content = self._call_llm_with_retry(prompt)
        if not content:
            return ClinicalContext()

        data = self._parse_json_response(content)
        if data:
            return ClinicalContext(**data)
        return ClinicalContext()

    def enrich_text(self, raw_text: str) -> PharmaContextOutput:
        """
        Main enrichment pipeline:
        1. Extract entities from raw text
        2. Verify against Knowledge Base (OpenFDA/RxNorm)
        3. Generate Clinical Context
        4. Return structured output
        """
        # Step 1: LLM Extraction
        extracted_entities = self._extract_entities_with_llm(raw_text)

        # Step 2: Verification
        search_term = (
            extracted_entities.brand_name or extracted_entities.generic_name
        )

        verification = VerificationData(confidence_score=0.0)

        if search_term:
            kb_result = self.kb.verify_drug(search_term)

            # Map OpenFDA results
            if kb_result["openfda"]:
                ofd = kb_result["openfda"]
                verification.openfda = OpenFDAResult(
                    is_fda_found=True,
                    brand_name=ofd.get("brand_name")
                    and ofd.get("brand_name")[0],
                    generic_name=ofd.get("generic_name")
                    and ofd.get("generic_name")[0],
                    manufacturer_name=ofd.get("manufacturer_name")
                    and ofd.get("manufacturer_name")[0],
                    product_ndc=ofd.get("product_ndc")
                    and ofd.get("product_ndc")[0],
                )
                verification.confidence_score += 0.4

            # Map RxNorm results
            if kb_result["rxnorm"]:
                rx = kb_result["rxnorm"]
                verification.rxnorm = RxNormResult(
                    is_rxnorm_found=True,
                    rxcui=rx.get("rxcui"),
                    name=rx.get("name"),
                )
                verification.confidence_score += 0.4

            # Boost confidence if key fields extracted
            if extracted_entities.brand_name and extracted_entities.strength:
                verification.confidence_score += 0.2

        # Step 3: Clinical Enrichment
        final_drug_name = search_term
        if verification.openfda.brand_name:
            final_drug_name = verification.openfda.brand_name

        clinical_context = ClinicalContext()
        if final_drug_name:
            clinical_context = self._generate_clinical_context(
                final_drug_name, {}
            )

        # Step 4: Final Output
        return PharmaContextOutput(
            raw_ocr_text=raw_text,
            extracted_entities=extracted_entities,
            verification=verification,
            clinical_enrichment=clinical_context,
            provenance={
                "search_term_used": search_term,
                "verified_name_used": final_drug_name,
                "llm_model": self.model_name,
            },
        )
