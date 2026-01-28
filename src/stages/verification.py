"""
Stage 2: Verification & Fuzzy Entity Resolution

Verifies extracted drug information against authoritative sources:
- OpenFDA Drug Labels
- RxNorm (NIH)
"""

import logging
from typing import Optional, Dict, List

import requests
from rapidfuzz import process, fuzz


logger = logging.getLogger(__name__)


class OpenFDAClient:
    """Client for the OpenFDA Drug Labels API."""

    BASE_URL = "https://api.fda.gov/drug/label.json"

    @staticmethod
    def search_drug(drug_name: str, limit: int = 1) -> Optional[Dict]:
        """
        Search OpenFDA drug labels by brand_name or generic_name.

        Args:
            drug_name: Drug name to search for
            limit: Maximum number of results

        Returns:
            First matching result or None
        """
        if not drug_name:
            return None

        query = f'openfda.brand_name:"{drug_name}"+OR+openfda.generic_name:"{drug_name}"'
        params = {"search": query, "limit": limit}

        try:
            response = requests.get(
                OpenFDAClient.BASE_URL, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenFDA API request failed for {drug_name}: {e}")
        return None


class RxNormClient:
    """Client for the RxNorm REST API."""

    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    @staticmethod
    def search_rxcui(name: str) -> Optional[Dict]:
        """
        Search for RxNorm Concept Unique Identifier (RxCUI) by name.
        Uses approximate matching.
        """
        if not name:
            return None

        url = f"{RxNormClient.BASE_URL}/approximateTerm.json"
        params = {"term": name, "maxEntries": 1}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if (
                "approximateGroup" in data
                and "candidate" in data["approximateGroup"]
            ):
                candidates = data["approximateGroup"]["candidate"]
                if candidates:
                    return candidates[0]
        except requests.exceptions.RequestException as e:
            logger.warning(f"RxNorm API request failed for {name}: {e}")
        return None

    @staticmethod
    def get_properties(rxcui: str) -> Optional[Dict]:
        """Get properties for a given RxCUI."""
        if not rxcui:
            return None

        url = f"{RxNormClient.BASE_URL}/rxcui/{rxcui}/allProperties.json"
        params = {"prop": "all"}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if (
                "propConceptGroup" in data
                and "propConcept" in data["propConceptGroup"]
            ):
                return data["propConceptGroup"]["propConcept"][0]
        except requests.exceptions.RequestException as e:
            logger.warning(f"RxNorm Properties failed for {rxcui}: {e}")
        return None


class KnowledgeBase:
    """
    Facade for interacting with knowledge sources (OpenFDA, RxNorm).
    Provides unified verification interface.
    """

    def __init__(self):
        self.openfda = OpenFDAClient()
        self.rxnorm = RxNormClient()

    def verify_drug(self, name: str) -> Dict:
        """
        Verifies a drug name against OpenFDA and RxNorm.

        Args:
            name: Drug name to verify

        Returns:
            Dictionary with results from both sources
        """
        result = {"openfda": None, "rxnorm": None, "query": name}

        # OpenFDA Search
        openfda_data = self.openfda.search_drug(name)
        if openfda_data:
            openfda_info = openfda_data.get("openfda", {})
            result["openfda"] = {
                "brand_name": openfda_info.get("brand_name"),
                "generic_name": openfda_info.get("generic_name"),
                "manufacturer_name": openfda_info.get("manufacturer_name"),
                "product_ndc": openfda_info.get("product_ndc"),
                "application_number": openfda_info.get("application_number"),
            }

        # RxNorm Search
        rxnorm_candidate = self.rxnorm.search_rxcui(name)
        if rxnorm_candidate:
            result["rxnorm"] = {
                "rxcui": rxnorm_candidate.get("rxcui"),
                "score": rxnorm_candidate.get("score"),
                "name": rxnorm_candidate.get("synonym")
                or rxnorm_candidate.get("name"),
            }

        return result

    @staticmethod
    def fuzzy_match_name(
        query: str, candidates: List[str], threshold: int = 80
    ) -> Optional[str]:
        """
        Returns the best match from candidates if score > threshold.
        Uses RapidFuzz for fast fuzzy matching.
        """
        match = process.extractOne(query, candidates, scorer=fuzz.ratio)
        if match:
            best_match, score, index = match
            if score >= threshold:
                return best_match
        return None
