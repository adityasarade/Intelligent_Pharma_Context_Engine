import sys
import os
from dotenv import load_dotenv

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.enrichment import EnrichmentEngine

def test_enrichment():
    print("Initializing Enrichment Engine...")
    try:
        engine = EnrichmentEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return

    # specific sample text mimicking a real label
    sample_text = """
    NDC 0093-3147-05
    AMOXICILLIN
    CAPSULES, USP
    500 mg
    Rx only
    500 CAPSULES
    TEVA
    """

    print("\n--- Input Text ---")
    print(sample_text)
    
    print("\n--- Running Enrichment ---")
    try:
        result = engine.enrich_text(sample_text)
        print("\n--- Result (JSON) ---")
        print(result.model_dump_json(indent=2))
        
        # Basic assertions
        assert result.verification.confidence_score > 0, "Confidence score should be > 0"
        assert result.extracted_entities.brand_name or result.extracted_entities.generic_name, "Should extract at least one name"
        print("\n[SUCCESS] Enrichment pipeline test passed!")
        
    except Exception as e:
        print(f"\n[FAILURE] Enrichment error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_dotenv()
    test_enrichment()
