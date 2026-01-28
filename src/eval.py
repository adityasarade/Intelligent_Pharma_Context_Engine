import logging
import rapidfuzz
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Eval")

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate (CER).
    CER = (S + D + I) / N
    Using Levenshtein distance.
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    
    dist = rapidfuzz.distance.Levenshtein.distance(reference, hypothesis)
    return dist / len(reference)

def calculate_entity_accuracy(ground_truth: Dict[str, str], prediction: Dict[str, str]) -> float:
    """
    Calculates accuracy of extracted entities.
    Simple exact match (case-insensitive) or very high fuzzy score.
    Returns percentage of matches.
    """
    matches = 0
    total = 0
    
    for key, gt_val in ground_truth.items():
        if gt_val is None: 
            continue # Skip optional fields if not in GT
        
        total += 1
        pred_val = prediction.get(key)
        
        if pred_val:
            # Fuzzy match to account for minor OCR typos
            ratio = rapidfuzz.fuzz.ratio(gt_val.lower(), pred_val.lower())
            if ratio > 90:
                matches += 1
            else:
                logger.debug(f"Mismatch [{key}]: GT='{gt_val}' | PRED='{pred_val}' (Score: {ratio})")
        else:
            logger.debug(f"Missing [{key}]: GT='{gt_val}'")
            
    return matches / total if total > 0 else 0.0

if __name__ == "__main__":
    # Smoke Test / Demo
    print("--- Evaluation Metrics Demo ---")
    
    # 1. CER Test
    ref_text = "AMOXICILLIN CAPSULES USP"
    hyp_text = "AMOXICILIN CAPSULES USP" # One 'L' missing
    cer = calculate_cer(ref_text, hyp_text)
    print(f"Reference:  {ref_text}")
    print(f"Hypothesis: {hyp_text}")
    print(f"CER: {cer:.4f}")
    
    # 2. Entity Test
    gt_entities = {
        "brand_name": "Amoxicillin",
        "strength": "500 mg",
        "manufacturer": "Teva"
    }
    pred_entities = {
        "brand_name": "Amoxcillin", # Typo
        "strength": "500 mg",
        "manufacturer": "Teva"
    }
    acc = calculate_entity_accuracy(gt_entities, pred_entities)
    print(f"\nEntity Accuracy: {acc:.2%}")
