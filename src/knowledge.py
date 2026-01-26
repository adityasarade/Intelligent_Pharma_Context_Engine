import requests
from typing import Optional, Dict, List, Tuple
import logging
from rapidfuzz import process, fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenFDAClient:
    BASE_URL = "https://api.fda.gov/drug/label.json"

    @staticmethod
    def search_drug(drug_name: str, limit: int = 1) -> Optional[Dict]:
        """
        Search OpenFDA drug labels by brand_name or generic_name.
        """
        if not drug_name:
            return None
        
        # Search in both brand_name and generic_name
        query = f'openfda.brand_name:"{drug_name}"+OR+openfda.generic_name:"{drug_name}"'
        params = {
            'search': query,
            'limit': limit
        }
        
        try:
            response = requests.get(OpenFDAClient.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                return data['results'][0]
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenFDA API request failed for {drug_name}: {e}")
            # Fallback or specific error handling can be added here
        return None

class RxNormClient:
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    @staticmethod
    def search_rxcui(name: str) -> Optional[Dict]:
        """
        Search for RxNorm Concept Unique Identifier (RxCUI) by name.
        Uses approximate matching.
        """
        if not name:
            return None

        # valid search via approximateTerm
        url = f"{RxNormClient.BASE_URL}/approximateTerm.json"
        params = {
            'term': name,
            'maxEntries': 1
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'approximateGroup' in data and 'candidate' in data['approximateGroup']:
                candidates = data['approximateGroup']['candidate']
                if candidates:
                    # Return the best candidate (first one)
                    return candidates[0]
        except requests.exceptions.RequestException as e:
            logger.warning(f"RxNorm API request failed for {name}: {e}")
        return None

    @staticmethod
    def get_properties(rxcui: str) -> Optional[Dict]:
        """
        Get properties for a given RxCUI.
        """
        if not rxcui:
            return None
            
        url = f"{RxNormClient.BASE_URL}/rxcui/{rxcui}/allProperties.json"
        params = {'prop': 'all'}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'propConceptGroup' in data and 'propConcept' in data['propConceptGroup']:
                 return data['propConceptGroup']['propConcept'][0] # Return first property set
        except requests.exceptions.RequestException as e:
            logger.warning(f"RxNorm Properties failed for {rxcui}: {e}")
        return None


class KnowledgeBase:
    """
    Facade for interacting with knowledge sources.
    """
    def __init__(self):
        self.openfda = OpenFDAClient()
        self.rxnorm = RxNormClient()

    def verify_drug(self, name: str) -> Dict:
        """
        Verifies a drug name against OpenFDA and RxNorm.
        Returns a dictionary with results from both.
        """
        result = {
            'openfda': None,
            'rxnorm': None,
            'query': name
        }
        
        # OpenFDA Search
        openfda_data = self.openfda.search_drug(name)
        if openfda_data:
            openfda_info = openfda_data.get('openfda', {})
            result['openfda'] = {
                'brand_name': openfda_info.get('brand_name'),
                'generic_name': openfda_info.get('generic_name'),
                'manufacturer_name': openfda_info.get('manufacturer_name'),
                'product_ndc': openfda_info.get('product_ndc'),
                'application_number': openfda_info.get('application_number')
            }
        
        # RxNorm Search
        rxnorm_candidate = self.rxnorm.search_rxcui(name)
        if rxnorm_candidate:
            result['rxnorm'] = {
                'rxcui': rxnorm_candidate.get('rxcui'),
                'score': rxnorm_candidate.get('score'),
                'name': rxnorm_candidate.get('synonym') or rxnorm_candidate.get('name') # usage depends on API response structure
            }

        return result

    @staticmethod
    def fuzzy_match_name(query: str, candidates: List[str], threshold: int = 80) -> Optional[str]:
        """
        Returns the best match from candidates if score > threshold.
        """
        match = process.extractOne(query, candidates, scorer=fuzz.ratio)
        if match:
            best_match, score, index = match
            if score >= threshold:
                return best_match
        return None
