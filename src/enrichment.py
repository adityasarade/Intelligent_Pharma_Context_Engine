import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time
from typing import Dict, Any, Optional
import logging

from src.schema import (
    PharmaContextOutput, 
    EntityExtraction, 
    VerificationData, 
    ClinicalContext, 
    OpenFDAResult, 
    RxNormResult
)
from src.knowledge import KnowledgeBase

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentEngine:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is missing.")
        
        genai.configure(api_key=self.api_key)
        self.model_name = 'gemini-2.0-flash'
        self.model = genai.GenerativeModel(self.model_name) 
        self.kb = KnowledgeBase()

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """
        Helper to call LLM with retry logic for rate limits (429).
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    logger.warning(f"Rate limit hit. Retrying in {(attempt + 1) * 5} seconds...")
                    time.sleep((attempt + 1) * 5)
                else:
                    logger.error(f"LLM call failed: {e}")
                    return None
        logger.error("Max retries exceeded for LLM call.")
        return None

    def _extract_entities_with_llm(self, text: str) -> EntityExtraction:
        """
        Uses Gemini to extract entities (Brand, Generic, Strength, etc.) from raw text.
        Returns an EntityExtraction object.
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

        # Simple cleanup to ensure JSON parsing
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
        
        try:
            data = json.loads(content)
            return EntityExtraction(**data)
        except json.JSONDecodeError:
            logger.error(f"LLM JSON Parse Error. Content: {content}")
            return EntityExtraction()

    def _generate_clinical_context(self, drug_name: str, verified_info: Dict) -> ClinicalContext:
        """
        Generates clinical context (indications, etc.) using Gemini, 
        grounded by the verified drug name.
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

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        try:
            data = json.loads(content)
            return ClinicalContext(**data)
        except json.JSONDecodeError:
            logger.error(f"LLM JSON Parse Error (Enrichment). Content: {content}")
            return ClinicalContext()

    def enrich_text(self, raw_text: str) -> PharmaContextOutput:
        """
        Main pipeline:
        1. Extract entities from raw text.
        2. Verify against Knowledge Base (OpenFDA/RxNorm).
        3. Generate/Enrich with Clinical Context.
        4. Return structured object.
        """
        # Step 1: LLM Extraction
        extracted_entities = self._extract_entities_with_llm(raw_text)
        
        # Step 2: Verification
        # Use brand name if available, else generic
        search_term = extracted_entities.brand_name or extracted_entities.generic_name
        
        verification = VerificationData(confidence_score=0.0)
        
        if search_term:
            kb_result = self.kb.verify_drug(search_term)
            
            # Map OpenFDA results
            if kb_result['openfda']:
                ofd = kb_result['openfda']
                verification.openfda = OpenFDAResult(
                    is_fda_found=True,
                    brand_name=ofd.get('brand_name') and ofd.get('brand_name')[0], # often lists
                    generic_name=ofd.get('generic_name') and ofd.get('generic_name')[0],
                    manufacturer_name=ofd.get('manufacturer_name') and ofd.get('manufacturer_name')[0],
                    product_ndc=ofd.get('product_ndc') and ofd.get('product_ndc')[0]
                )
                verification.confidence_score += 0.4
            
            # Map RxNorm results
            if kb_result['rxnorm']:
                rx = kb_result['rxnorm']
                verification.rxnorm = RxNormResult(
                    is_rxnorm_found=True,
                    rxcui=rx.get('rxcui'),
                    name=rx.get('name')
                )
                verification.confidence_score += 0.4
                
            # Basic confidence logic
            if extracted_entities.brand_name and extracted_entities.strength:
                verification.confidence_score += 0.2

        # Step 3: Clinical Enrichment
        # Use verified name if possible, else extracted
        final_drug_name = search_term
        if verification.openfda.brand_name:
            final_drug_name = verification.openfda.brand_name
        
        clinical_context = ClinicalContext()
        if final_drug_name:
            clinical_context = self._generate_clinical_context(final_drug_name, {})

        # Step 4: Final Output
        return PharmaContextOutput(
            raw_ocr_text=raw_text,
            extracted_entities=extracted_entities,
            verification=verification,
            clinical_enrichment=clinical_context,
            provenance={
                "search_term_used": search_term,
                "verified_name_used": final_drug_name,
                "llm_model": self.model_name
            }
        )

if __name__ == "__main__":
    # Simple test
    engine = EnrichmentEngine()
    test_text = "NOC 0000-0000-00\nAMOXICILLIN\nCapsules, USP\n500 mg\nRx only"
    result = engine.enrich_text(test_text)
    print(result.model_dump_json(indent=2))
