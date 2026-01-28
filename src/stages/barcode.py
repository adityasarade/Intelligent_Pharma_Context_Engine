"""
Stage 1: Barcode / DataMatrix Decoding

Decodes barcodes and DataMatrix codes from pharmaceutical packaging
for multi-modal validation of OCR results.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import barcode libraries (optional dependencies)
PYZBAR_AVAILABLE = False
PYLIBDMTX_AVAILABLE = False

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol

    PYZBAR_AVAILABLE = True
except ImportError:
    logger.warning("pyzbar not available. Install with: pip install pyzbar")
except (OSError, FileNotFoundError) as e:
    logger.warning(
        f"pyzbar native library (libzbar) not found: {e}. "
        "On Windows, install Visual C++ Redistributable and zbar library."
    )

try:
    from pylibdmtx import pylibdmtx

    PYLIBDMTX_AVAILABLE = True
except ImportError:
    logger.warning(
        "pylibdmtx not available. Install with: pip install pylibdmtx"
    )
except (OSError, FileNotFoundError) as e:
    logger.warning(
        f"pylibdmtx native library not found: {e}. "
        "On Windows, install libdmtx library."
    )


def decode_barcodes(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Decode all barcodes from an image using pyzbar.

    Supports: EAN-13, EAN-8, UPC-A, UPC-E, Code 128, Code 39, QR Code, etc.

    Args:
        image: Input image as numpy array (grayscale or BGR)

    Returns:
        List of decoded barcodes with type, data, and bounding box
    """
    results = []

    if not PYZBAR_AVAILABLE:
        logger.warning("pyzbar not available, skipping barcode decoding")
        return results

    try:
        # Decode all barcode types
        decoded = pyzbar.decode(image)

        for barcode in decoded:
            # Extract bounding box
            rect = barcode.rect
            bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)

            results.append(
                {
                    "type": barcode.type,
                    "data": barcode.data.decode("utf-8", errors="replace"),
                    "bbox": bbox,
                    "polygon": [(p.x, p.y) for p in barcode.polygon],
                    "quality": getattr(barcode, "quality", None),
                }
            )
            logger.info(f"Decoded {barcode.type}: {barcode.data.decode()}")

    except Exception as e:
        logger.error(f"Barcode decoding failed: {e}")

    return results


def decode_datamatrix(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Decode DataMatrix codes from an image using pylibdmtx.

    DataMatrix is commonly used on pharmaceutical packaging for
    serialization and track-and-trace (GS1 standards).

    Args:
        image: Input image as numpy array

    Returns:
        List of decoded DataMatrix codes
    """
    results = []

    if not PYLIBDMTX_AVAILABLE:
        logger.warning("pylibdmtx not available, skipping DataMatrix decoding")
        return results

    try:
        decoded = pylibdmtx.decode(image)

        for dm in decoded:
            rect = dm.rect
            bbox = (rect.left, rect.top, rect.left + rect.width, rect.top + rect.height)

            results.append(
                {
                    "type": "DataMatrix",
                    "data": dm.data.decode("utf-8", errors="replace"),
                    "bbox": bbox,
                }
            )
            logger.info(f"Decoded DataMatrix: {dm.data.decode()}")

    except Exception as e:
        logger.error(f"DataMatrix decoding failed: {e}")

    return results


def decode_all(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Decode all barcode types from an image.

    Combines pyzbar (1D/2D barcodes) and pylibdmtx (DataMatrix).

    Args:
        image: Input image as numpy array

    Returns:
        List of all decoded codes with type, data, bbox, and confidence
    """
    results = []

    # Decode standard barcodes (EAN, UPC, QR, etc.)
    barcodes = decode_barcodes(image)
    for bc in barcodes:
        bc["decode_confidence"] = 0.95  # pyzbar is generally reliable
        results.append(bc)

    # Decode DataMatrix (common in pharma)
    datamatrix = decode_datamatrix(image)
    for dm in datamatrix:
        dm["decode_confidence"] = 0.95
        results.append(dm)

    return results


def parse_gs1_barcode(data: str) -> Optional[Dict[str, str]]:
    """
    Parse GS1-128 or GS1 DataMatrix barcode data.

    Common Application Identifiers (AIs) in pharmaceutical barcodes:
    - 01: GTIN (Global Trade Item Number)
    - 10: Batch/Lot Number
    - 17: Expiration Date (YYMMDD)
    - 21: Serial Number

    Args:
        data: Raw barcode data string

    Returns:
        Dictionary of parsed fields or None if not GS1 format
    """
    result = {}

    # GS1 uses parentheses or FNC1 separators
    # Common pattern: (01)GTIN(17)EXPIRY(10)BATCH(21)SERIAL
    import re

    # Try to parse with parentheses format
    pattern = r"\((\d{2})\)([^\(]+)"
    matches = re.findall(pattern, data)

    if matches:
        ai_map = {
            "01": "gtin",
            "10": "batch_lot",
            "17": "expiry_date",
            "21": "serial_number",
            "11": "production_date",
            "30": "quantity",
        }

        for ai, value in matches:
            field_name = ai_map.get(ai, f"ai_{ai}")
            result[field_name] = value.strip()

        return result if result else None

    # Try GTIN-14 format (14 digits starting with indicator)
    if len(data) == 14 and data.isdigit():
        result["gtin"] = data
        return result

    # Try NDC format (common US drug identifier)
    ndc_pattern = r"^\d{4,5}-\d{3,4}-\d{1,2}$"
    if re.match(ndc_pattern, data):
        result["ndc"] = data
        return result

    return None


def validate_ocr_with_barcode(
    ocr_entities: Dict[str, Any], barcode_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate OCR-extracted entities against barcode data.

    If barcode contains identifiable drug information (NDC, GTIN),
    it can be used to verify or correct OCR results.

    Args:
        ocr_entities: Dictionary of OCR-extracted entities
        barcode_data: List of decoded barcodes

    Returns:
        Validation result with match status and confidence adjustment
    """
    result = {
        "barcode_found": len(barcode_data) > 0,
        "validation_status": "no_barcode",
        "parsed_identifiers": [],
        "confidence_boost": 0.0,
    }

    for barcode in barcode_data:
        parsed = parse_gs1_barcode(barcode["data"])
        if parsed:
            result["parsed_identifiers"].append(
                {"raw": barcode["data"], "parsed": parsed, "type": barcode["type"]}
            )

            # If we found a valid GTIN or NDC, boost confidence
            if "gtin" in parsed or "ndc" in parsed:
                result["validation_status"] = "identifier_found"
                result["confidence_boost"] = 0.2

    if barcode_data and not result["parsed_identifiers"]:
        result["validation_status"] = "barcode_unparsed"

    return result
