"""
Stage 1: OCR / Text Recognition

Extracts text from preprocessed images using Tesseract OCR.
"""

import os
from typing import List, Dict, Any, Union

import numpy as np
import pytesseract
from PIL import Image


def _get_tesseract_cmd():
    """Get Tesseract command path from environment variable."""
    return os.getenv("TESSERACT_CMD", None)


# Configure Tesseract path if specified
if _get_tesseract_cmd():
    pytesseract.pytesseract.tesseract_cmd = _get_tesseract_cmd()


def run_ocr(image: Union[np.ndarray, Image.Image], lang: str = "eng") -> str:
    """
    Runs Tesseract OCR on the given image.

    Args:
        image: Input image (numpy array or PIL Image)
        lang: Language code for OCR

    Returns:
        Extracted text string
    """
    # Configuration:
    # --oem 3: Default, based on what causes the best results (LSTM usually)
    # --psm 6: Assume a single uniform block of text (good for cropped labels)
    config = r"--oem 3 --psm 6"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    text = pytesseract.image_to_string(image, lang=lang, config=config)
    return text.strip()


def run_ocr_with_data(
    image: Union[np.ndarray, Image.Image], lang: str = "eng"
) -> Dict[str, Any]:
    """
    Runs Tesseract and returns detailed data including bounding boxes and confidences.
    """
    config = r"--oem 3 --psm 6"

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    data = pytesseract.image_to_data(
        image, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    return data


def extract_text_blocks(
    image: np.ndarray, min_conf: int = 40
) -> List[Dict[str, Any]]:
    """
    Runs OCR and filters for confident text blocks.

    Args:
        image: Input image as numpy array
        min_conf: Minimum confidence threshold (0-100)

    Returns:
        List of text blocks with text, confidence, bbox, block_num, line_num
    """
    data = run_ocr_with_data(image)
    blocks = []

    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if conf > min_conf and len(text) > 1:
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            blocks.append(
                {
                    "text": text,
                    "conf": conf,
                    "bbox": (x, y, x + w, y + h),
                    "block_num": data["block_num"][i],
                    "line_num": data["line_num"][i],
                }
            )

    return blocks
