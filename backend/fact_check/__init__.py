"""
Fact-Check Module
Comprehensive fact-checking pipeline with AI-powered analysis.
"""
from .fact_checker import FactChecker
from .input_classifier import InputClassifier
from .content_extractor import ContentExtractor
from .web_searcher import WebSearcher
from .credibility_manager import CredibilityManager
from .feedback_handler import FeedbackHandler
from .temporal_analyzer import TemporalAnalyzer

__all__ = [
    'FactChecker',
    'InputClassifier', 
    'ContentExtractor',
    'WebSearcher',
    'CredibilityManager',
    'FeedbackHandler',
    'TemporalAnalyzer'
]
__version__ = '1.0.0'
