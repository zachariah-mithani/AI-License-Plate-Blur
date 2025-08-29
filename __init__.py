"""
License Plate Blur - ALPR Privacy Protection System
Advanced computer vision application for privacy protection
"""

__version__ = "1.0.0"
__author__ = "Zachariah Mithani"
__description__ = "AI-powered license plate detection and privacy protection system"

from .detector import LicensePlateDetector
from .privacy import PrivacyProtector, BlurMode
from .video_processor import VideoProcessor

__all__ = [
    "LicensePlateDetector", 
    "PrivacyProtector", 
    "BlurMode", 
    "VideoProcessor"
]