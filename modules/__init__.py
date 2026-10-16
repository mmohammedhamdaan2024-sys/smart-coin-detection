"""
Modules package for Smart Coin Detection System.
Provides functional image processing, segmentation, measurement, classification,
condition screening, report generation, and database interfaces.
"""

from .image_loader import ImageLoader
from .preprocessing import PreprocessingPipeline
from .segmentation import CoinSegmenter
from .measurements import GeometricAnalyzer
from .reference_database import ReferenceDatabase
from .coin_classifier import CoinClassifier
from .condition_analyzer import ConditionAnalyzer
from .weight_analysis import WeightEstimator
from .age_analysis import AgeEstimator
from .coin_detector import CoinDetector
from .report_generator import ReportGenerator
from .history_manager import HistoryManager

__all__ = [
    "ImageLoader",
    "PreprocessingPipeline",
    "CoinSegmenter",
    "GeometricAnalyzer",
    "ReferenceDatabase",
    "CoinClassifier",
    "ConditionAnalyzer",
    "WeightEstimator",
    "AgeEstimator",
    "CoinDetector",
    "ReportGenerator",
    "HistoryManager",
]
