"""
Intelligent Pharma-Context Engine

An end-to-end prototype for extracting structured metadata from
pharmaceutical packaging images.
"""

from .pipeline import PharmaContextPipeline
from .models.schema import PharmaContextOutput

__version__ = "0.1.0"

__all__ = ["PharmaContextPipeline", "PharmaContextOutput", "__version__"]
