"""Tests for the preprocessing stage."""

import pytest
import numpy as np

from src.stages.preprocessing import (
    to_grayscale,
    apply_clahe,
    apply_bilateral_filter,
)


class TestPreprocessing:
    """Test suite for preprocessing functions."""

    def test_to_grayscale(self):
        """Test grayscale conversion."""
        # Create a dummy BGR image
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 2] = 255  # Red channel

        gray = to_grayscale(bgr_image)

        assert gray.ndim == 2
        assert gray.shape == (100, 100)

    def test_apply_clahe(self):
        """Test CLAHE enhancement."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        enhanced = apply_clahe(gray_image)

        assert enhanced.shape == gray_image.shape
        assert enhanced.dtype == np.uint8

    def test_apply_bilateral_filter(self):
        """Test bilateral filtering."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        filtered = apply_bilateral_filter(image)

        assert filtered.shape == image.shape
