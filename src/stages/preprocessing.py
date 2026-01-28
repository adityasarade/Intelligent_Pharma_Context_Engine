"""
Stage 1: Image Preprocessing

Applies contrast enhancement, denoising, and shadow removal
to prepare images for OCR.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def load_image(image_path: str) -> np.ndarray:
    """Loads an image from a file path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_clahe(
    gray_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Applies bilateral filtering to reduce noise while preserving edges."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """
    Removes shadows/glare using morphological operations.
    Helps with uneven lighting on pharmaceutical labels.
    """
    rgb_planes = cv2.split(image)
    result_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8UC1,
        )
        result_planes.append(norm_img)

    return cv2.merge(result_planes)


def preprocess_pipeline(
    image_path: str, debug_output: bool = False
) -> Dict[str, np.ndarray]:
    """
    Runs the full Stage 1 image preprocessing pipeline.

    Returns a dictionary of intermediate images for traceability:
    - original: The raw loaded image
    - normalized: Shadow/illumination corrected
    - denoised: After bilateral filtering
    - gray: Grayscale version
    - enhanced_gray: Final CLAHE-enhanced image (primary OCR input)
    """
    original = load_image(image_path)

    # 1. Shadow/Illumination correction (on color image)
    normalized = remove_shadows(original)

    # 2. Denoising
    denoised = apply_bilateral_filter(normalized)

    # 3. Grayscale
    gray = to_grayscale(denoised)

    # 4. CLAHE
    enhanced = apply_clahe(gray)

    return {
        "original": original,
        "normalized": normalized,
        "denoised": denoised,
        "gray": gray,
        "enhanced_gray": enhanced,
    }
