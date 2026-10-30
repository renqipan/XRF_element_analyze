"""XRF spectrum analysis using jointly fitted element templates."""
__version__ = "0.2.0"

from .config import AnalysisConfig
from .pipeline import analyze

__all__ = ["AnalysisConfig", "analyze"]
