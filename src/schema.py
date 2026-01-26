from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class EntityExtraction(BaseModel):
    brand_name: Optional[str] = Field(None, description="The brand name of the medication")
    generic_name: Optional[str] = Field(None, description="The generic name of the medication")
    strength: Optional[str] = Field(None, description="The strength of the medication (e.g., 500mg, 10mg/mL)")
    dosage_form: Optional[str] = Field(None, description="The form of the medication (e.g., Tablet, Capsule, Solution)")
    manufacturer: Optional[str] = Field(None, description="The manufacturer of the medication")

class OpenFDAResult(BaseModel):
    application_number: Optional[str] = None
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    manufacturer_name: Optional[str] = None
    product_ndc: Optional[str] = None
    is_fda_found: bool = False

class RxNormResult(BaseModel):
    rxcui: Optional[str] = None
    name: Optional[str] = None
    synonym: Optional[str] = None
    tty: Optional[str] = None  # Term Type
    is_rxnorm_found: bool = False

class VerificationData(BaseModel):
    openfda: OpenFDAResult = Field(default_factory=OpenFDAResult)
    rxnorm: RxNormResult = Field(default_factory=RxNormResult)
    confidence_score: float = Field(..., description="Overall confidence score of the verification (0.0 to 1.0)")
    notes: List[str] = Field(default_factory=list, description="Verification notes or warnings")

class ClinicalContext(BaseModel):
    indications: List[str] = Field(default_factory=list, description="What the drug is used for")
    contraindications: List[str] = Field(default_factory=list, description="When the drug should not be used")
    warnings: List[str] = Field(default_factory=list, description="Major warnings and precautions")
    side_effects: List[str] = Field(default_factory=list, description="Common side effects")

class PharmaContextOutput(BaseModel):
    raw_ocr_text: str
    extracted_entities: EntityExtraction
    verification: VerificationData
    clinical_enrichment: ClinicalContext
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Source of information and audit trail")
