"""Tests for evaluation metrics."""

import pytest

from src.evaluation.metrics import calculate_cer, calculate_entity_accuracy


class TestCER:
    """Test suite for Character Error Rate calculation."""

    def test_cer_perfect_match(self):
        """Test CER with identical strings."""
        cer = calculate_cer("AMOXICILLIN", "AMOXICILLIN")
        assert cer == 0.0

    def test_cer_one_char_error(self):
        """Test CER with single character difference."""
        cer = calculate_cer("AMOXICILLIN", "AMOXICILIN")  # Missing L
        assert 0.0 < cer < 0.2

    def test_cer_empty_reference(self):
        """Test CER with empty reference."""
        assert calculate_cer("", "text") == 1.0
        assert calculate_cer("", "") == 0.0


class TestEntityAccuracy:
    """Test suite for Entity Match Rate calculation."""

    def test_entity_accuracy_perfect(self):
        """Test with all fields matching."""
        gt = {"brand_name": "Amoxicillin", "strength": "500 mg"}
        pred = {"brand_name": "Amoxicillin", "strength": "500 mg"}

        acc = calculate_entity_accuracy(gt, pred)
        assert acc == 1.0

    def test_entity_accuracy_partial(self):
        """Test with partial match."""
        gt = {"brand_name": "Amoxicillin", "strength": "500 mg"}
        pred = {"brand_name": "Amoxicillin", "strength": "250 mg"}

        acc = calculate_entity_accuracy(gt, pred)
        assert acc == 0.5

    def test_entity_accuracy_fuzzy(self):
        """Test fuzzy matching for minor typos."""
        gt = {"brand_name": "Amoxicillin"}
        pred = {"brand_name": "Amoxicilin"}  # Minor typo

        acc = calculate_entity_accuracy(gt, pred)
        assert acc == 1.0  # Should still match due to >90% similarity
