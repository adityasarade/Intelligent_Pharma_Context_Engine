from src.knowledge import KnowledgeBase
import json

def test_knowledge():
    print("Initializing KnowledgeBase...")
    kb = KnowledgeBase()
    
    drug = "Amoxicillin"
    print(f"Verifying {drug}...")
    try:
        result = kb.verify_drug(drug)
        print(json.dumps(result, indent=2))
        
        # Assertions
        assert result['openfda'] is not None, "OpenFDA should find Amoxicillin"
        assert result['rxnorm'] is not None, "RxNorm should find Amoxicillin"
        print("[SUCCESS] KnowledgeBase verification passed!")
    except Exception as e:
        print(f"[FAILURE] KnowledgeBase error: {e}")

if __name__ == "__main__":
    test_knowledge()
