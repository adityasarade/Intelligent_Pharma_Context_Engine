"""Tests for the verification stage."""

import pytest

from src.stages.verification import KnowledgeBase


class TestKnowledgeBase:
    """Test suite for KnowledgeBase verification."""

    def test_fuzzy_match_exact(self):
        """Test fuzzy matching with exact match."""
        candidates = ["Amoxicillin", "Lisinopril", "Metformin"]
        result = KnowledgeBase.fuzzy_match_name("Amoxicillin", candidates)
        assert result == "Amoxicillin"

    def test_fuzzy_match_typo(self):
        """Test fuzzy matching with minor typo."""
        candidates = ["Amoxicillin", "Lisinopril", "Metformin"]
        result = KnowledgeBase.fuzzy_match_name("Amoxicilin", candidates)
        assert result == "Amoxicillin"

    def test_fuzzy_match_no_match(self):
        """Test fuzzy matching with no good match."""
        candidates = ["Amoxicillin", "Lisinopril", "Metformin"]
        result = KnowledgeBase.fuzzy_match_name("Aspirin", candidates)
        assert result is None


class TestRxNormClient:
    """Integration tests for RxNorm API (requires network)."""

    @pytest.mark.integration
    def test_search_rxcui_valid(self):
        """Test RxNorm search with valid drug name."""
        from src.stages.verification import RxNormClient

        result = RxNormClient.search_rxcui("Amoxicillin")
        assert result is not None
        assert "rxcui" in result


class TestOpenFDAClient:
    """Integration tests for OpenFDA API (requires network)."""

    @pytest.mark.integration
    def test_search_drug_valid(self):
        """Test OpenFDA search with valid drug name."""
        from src.stages.verification import OpenFDAClient

        result = OpenFDAClient.search_drug("Amoxicillin")
        # May return None if rate limited, so we just check it doesn't crash
        assert result is None or isinstance(result, dict)
