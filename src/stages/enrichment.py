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
from datetime import datetime
from typing import Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from ..models.schema import (
    EntityExtraction,
    ClinicalContext,
    OpenFDAResult,
    RxNormResult,
    VerificationData,
    CandidateField,
    MatchEvidence,
    Provenance,
)
from .verification import KnowledgeBase


load_dotenv()
logger = logging.getLogger(__name__)


class EnrichmentEngine:
    """
    Orchestrates entity extraction and clinical enrichment using Gemini LLM.
    """

    CONFIDENCE_THRESHOLD = 0.6  # Below this, flag for human review

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
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.error(f"LLM JSON Parse Error. Content: {content[:200]}")
            return None

    def extract_entities(self, text: str) -> EntityExtraction:
        """
        Uses Gemini to extract entities from raw OCR text.

        Extracts: drug_name, generic_name, manufacturer, composition,
                  strength, dosage_form, batch_lot, expiry_date
        """
        prompt = f"""You are an expert pharmaceutical entity extractor.
Extract the following fields from the provided OCR text of a medicine label/bottle:

- drug_name: The brand/trade name of the medication
- generic_name: The generic/scientific name
- manufacturer: The company that makes it
- composition: Active ingredients (e.g., "Amoxicillin Trihydrate 500mg")
- strength: The dosage strength (e.g., "500mg", "10mg/mL")
- dosage_form: The form (e.g., "Tablet", "Capsule", "Syrup")
- batch_lot: Batch or Lot number if visible
- expiry_date: Expiration date if visible

Return a valid JSON object:
{{
    "drug_name": "...",
    "generic_name": "...",
    "manufacturer": "...",
    "composition": "...",
    "strength": "...",
    "dosage_form": "...",
    "batch_lot": "...",
    "expiry_date": "..."
}}

Rules:
- If a field is not found, set it to null
- Do not hallucinate or invent information
- Extract exactly what is visible in the text

OCR Text:
{text}
"""
        content = self._call_llm_with_retry(prompt)
        if not content:
            return EntityExtraction()

        data = self._parse_json_response(content)
        if data:
            # Map brand_name to drug_name for compatibility
            if "brand_name" in data and "drug_name" not in data:
                data["drug_name"] = data.pop("brand_name")
            return EntityExtraction(**data)
        return EntityExtraction()

    def verify_entities(
        self,
        entities: EntityExtraction,
        barcode_validation: Optional[Dict[str, Any]] = None,
    ) -> VerificationData:
        """
        Verify extracted entities against OpenFDA and RxNorm.
        """
        verification = VerificationData()
        methods_used = []
        confidence = 0.0

        # Determine search term
        search_term = entities.drug_name or entities.generic_name
        if not search_term:
            verification.human_review_needed = True
            verification.review_hint = "No drug name extracted from OCR"
            return verification

        # Query knowledge bases
        kb_result = self.kb.verify_drug(search_term)

        # Process OpenFDA results
        if kb_result["openfda"]:
            ofd = kb_result["openfda"]
            verification.openfda = OpenFDAResult(
                is_found=True,
                brand_name=(
                    ofd.get("brand_name")[0]
                    if ofd.get("brand_name")
                    else None
                ),
                generic_name=(
                    ofd.get("generic_name")[0]
                    if ofd.get("generic_name")
                    else None
                ),
                manufacturer_name=(
                    ofd.get("manufacturer_name")[0]
                    if ofd.get("manufacturer_name")
                    else None
                ),
                product_ndc=(
                    ofd.get("product_ndc")[0]
                    if ofd.get("product_ndc")
                    else None
                ),
            )
            confidence += 0.35
            methods_used.append("openfda_lookup")

        # Process RxNorm results
        if kb_result["rxnorm"]:
            rx = kb_result["rxnorm"]
            verification.rxnorm = RxNormResult(
                is_found=True,
                rxcui=rx.get("rxcui"),
                name=rx.get("name"),
            )
            confidence += 0.35
            methods_used.append("rxnorm_lookup")

        # Add candidate field tracking
        if search_term:
            match_method = "exact" if confidence > 0.5 else "fuzzy_edit"
            verification.candidate_fields["drug_name"] = CandidateField(
                text=search_term,
                normalized=search_term.lower().strip(),
                match=MatchEvidence(
                    method=match_method,
                    confidence=confidence,
                    source_text=search_term,
                    matched_text=verification.openfda.brand_name
                    or verification.rxnorm.name,
                ),
                verified=confidence > 0.5,
            )

        # Barcode validation boost
        if barcode_validation and barcode_validation.get("barcode_found"):
            verification.barcode_validation = barcode_validation
            if barcode_validation.get("validation_status") == "identifier_found":
                confidence += barcode_validation.get("confidence_boost", 0.2)
                methods_used.append("barcode")

        # Boost for complete extraction
        if entities.drug_name and entities.strength:
            confidence += 0.1

        # Cap confidence at 1.0
        verification.confidence_score = min(confidence, 1.0)
        verification.match_method = methods_used

        # Flag for human review if low confidence
        if verification.confidence_score < self.CONFIDENCE_THRESHOLD:
            verification.human_review_needed = True
            verification.review_hint = (
                f"Low confidence ({verification.confidence_score:.2f}). "
                "Manual verification recommended."
            )

        return verification

    def generate_clinical_context(
        self, drug_name: str, verified_info: Optional[Dict] = None
    ) -> ClinicalContext:
        """
        Generates clinical context using Gemini, grounded by verified drug name.
        """
        prompt = f"""Provide a brief, professional clinical summary for the drug: "{drug_name}".

Include:
1. Indications (List of 2-4 main uses)
2. Contraindications (List of 2-3 when NOT to use)
3. Warnings (2-3 major precautions)
4. Common Side Effects (3-5 most common)
5. Storage requirements (brief, e.g., "Store below 25°C")

Format as JSON:
{{
    "indications": ["..."],
    "contraindications": ["..."],
    "warnings": ["..."],
    "side_effects": ["..."],
    "storage": "..."
}}

Rules:
- Be concise and factual
- If unsure about any field, return an empty list or null
- Do not invent medical information
"""
        content = self._call_llm_with_retry(prompt)
        if not content:
            return ClinicalContext()

        data = self._parse_json_response(content)
        if data:
            return ClinicalContext(**data)
        return ClinicalContext()

    def generate_human_summary(
        self, drug_name: str, clinical: ClinicalContext
    ) -> str:
        """
        Generate a layperson-friendly summary of the drug information.
        """
        prompt = f"""Based on the following clinical information for "{drug_name}",
write a 2-3 sentence layperson-friendly summary covering:
- What the drug is used for
- Key storage instruction
- One or two common side effects to be aware of

Clinical data:
- Indications: {clinical.indications}
- Storage: {clinical.storage}
- Side effects: {clinical.side_effects}

Keep it simple, clear, and under 100 words. Do not add any information not provided above.
"""
        content = self._call_llm_with_retry(prompt)
        return content.strip() if content else ""

    def create_provenance(
        self, search_term: Optional[str], final_drug_name: Optional[str]
    ) -> Provenance:
        """Create provenance/audit trail for the enrichment."""
        return Provenance(
            openfda_query=search_term,
            openfda_fetch_datetime=datetime.utcnow() if search_term else None,
            rxnorm_query=search_term,
            rxnorm_fetch_datetime=datetime.utcnow() if search_term else None,
            llm_model=self.model_name,
            llm_call_datetime=datetime.utcnow(),
        )
