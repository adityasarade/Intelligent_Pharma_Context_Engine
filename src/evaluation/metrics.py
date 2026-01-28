"""
Evaluation metrics for the Pharma Context Engine.

Implements CER (Character Error Rate) and EMR (Entity Match Rate) as
specified in the technical specification.
"""

import logging
from typing import Dict

import rapidfuzz


logger = logging.getLogger(__name__)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate (CER).

    CER = (Substitutions + Deletions + Insertions) / Total Characters
    Uses Levenshtein distance.

    Args:
        reference: Ground truth text
        hypothesis: OCR output text

    Returns:
        CER score (0.0 = perfect, 1.0 = all wrong)
    """
    if not reference:
        return 1.0 if hypothesis else 0.0

    dist = rapidfuzz.distance.Levenshtein.distance(reference, hypothesis)
    return dist / len(reference)


def calculate_entity_accuracy(
    ground_truth: Dict[str, str], prediction: Dict[str, str]
) -> float:
    """
    Calculates accuracy of extracted entities (Entity Match Rate).

    Uses fuzzy matching (>90% similarity) to account for minor OCR typos.

    Args:
        ground_truth: Dictionary of field_name -> expected_value
        prediction: Dictionary of field_name -> predicted_value

    Returns:
        Accuracy as a percentage (0.0 to 1.0)
    """
    matches = 0
    total = 0

    for key, gt_val in ground_truth.items():
        if gt_val is None:
            continue  # Skip optional fields not in ground truth

        total += 1
        pred_val = prediction.get(key)

        if pred_val:
            # Fuzzy match to account for minor OCR typos
            ratio = rapidfuzz.fuzz.ratio(gt_val.lower(), pred_val.lower())
            if ratio > 90:
                matches += 1
            else:
                logger.debug(
                    f"Mismatch [{key}]: GT='{gt_val}' | PRED='{pred_val}' "
                    f"(Score: {ratio})"
                )
        else:
            logger.debug(f"Missing [{key}]: GT='{gt_val}'")

    return matches / total if total > 0 else 0.0
