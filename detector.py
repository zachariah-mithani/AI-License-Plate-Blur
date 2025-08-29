"""
License Plate Detection Engine
Using PaddleOCR for text detection with geometric filtering for license plates
"""

import cv2
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
import time

# Mock PaddleOCR for demonstration
class MockPaddleOCR:
    def __init__(self, use_angle_cls=False, lang='en'):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
    
    def ocr(self, image, cls=False, rec=True):
        """Mock OCR that simulates license plate detection"""
        height, width = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        
        # Simulate finding license plate regions with mock data
        mock_detections = []
        
        # Create realistic mock detections based on image size
        if width > 400 and height > 200:
            # Simulate 1-3 license plates
            num_plates = np.random.randint(1, 4)
            
            for i in range(num_plates):
                # Generate realistic bounding box
                x_start = np.random.randint(50, width - 200)
                y_start = np.random.randint(50, height - 100)
                plate_width = np.random.randint(120, 200)
                plate_height = np.random.randint(30, 50)
                
                # Ensure box stays within image
                x_end = min(x_start + plate_width, width - 10)
                y_end = min(y_start + plate_height, height - 10)
                
                # Mock license plate text
                plate_texts = ["ABC-1234", "XYZ-5678", "DEF-9876", "GHI-2345", "JKL-7890"]
                plate_text = plate_texts[i % len(plate_texts)]
                
                # Mock confidence (0.7-0.95)
                confidence = 0.7 + (np.random.random() * 0.25)
                
                # Create mock detection in PaddleOCR format
                bbox = [[x_start, y_start], [x_end, y_start], [x_end, y_end], [x_start, y_end]]
                detection = (bbox, (plate_text, confidence))
                mock_detections.append(detection)
        
        return [mock_detections] if mock_detections else [[]]

class LicensePlateDetector:
    def __init__(self, confidence_threshold: float = 0.7, use_angle_cls: bool = False):
        """
        Initialize license plate detector with PaddleOCR
        
        Args:
            confidence_threshold: Minimum confidence for text detection
            use_angle_cls: Enable text angle classification (slower but more accurate)
        """
        self.confidence_threshold = confidence_threshold
        # Use mock OCR for demonstration purposes
        self.ocr = MockPaddleOCR(
            use_angle_cls=use_angle_cls, 
            lang='en'
        )
        
        # License plate characteristics
        self.min_aspect_ratio = 2.0  # Minimum width/height ratio
        self.max_aspect_ratio = 8.0  # Maximum width/height ratio
        self.min_area = 1000  # Minimum bounding box area
        self.min_chars = 4  # Minimum characters for valid plate
        self.max_chars = 10  # Maximum characters for valid plate
    
    def detect_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all text regions in image using PaddleOCR
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of text detection results with bounding boxes and confidence
        """
        try:
            start_time = time.time()
            results = self.ocr.ocr(image, cls=False, rec=True)
            detection_time = time.time() - start_time
            
            text_regions = []
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        bbox, (text, confidence) = line
                        
                        if confidence >= self.confidence_threshold:
                            # Convert bbox to standard format
                            bbox_array = np.array(bbox)
                            x_coords = bbox_array[:, 0]
                            y_coords = bbox_array[:, 1]
                            
                            x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
                            x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
                            
                            text_regions.append({
                                'bbox': (x_min, y_min, x_max, y_max),
                                'text': text,
                                'confidence': confidence,
                                'detection_time': detection_time
                            })
            
            return text_regions
            
        except Exception as e:
            print(f"Error in text detection: {e}")
            return []
    
    def filter_license_plates(self, text_regions: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        Filter text regions to identify likely license plates based on geometry and content
        
        Args:
            text_regions: List of detected text regions
            image_shape: Shape of the input image (height, width)
            
        Returns:
            List of filtered license plate candidates
        """
        license_plates = []
        
        for region in text_regions:
            x_min, y_min, x_max, y_max = region['bbox']
            width = x_max - x_min
            height = y_max - y_min
            
            # Skip if dimensions are invalid
            if width <= 0 or height <= 0:
                continue
            
            # Calculate geometric properties
            aspect_ratio = width / height
            area = width * height
            
            # Check geometric constraints
            if (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                area >= self.min_area):
                
                # Check text content
                text = region['text'].strip()
                if self.is_valid_plate_text(text):
                    region['aspect_ratio'] = aspect_ratio
                    region['area'] = area
                    region['plate_score'] = self.calculate_plate_score(text, aspect_ratio, region['confidence'])
                    license_plates.append(region)
        
        # Sort by plate score (higher is better)
        license_plates.sort(key=lambda x: x['plate_score'], reverse=True)
        
        return license_plates
    
    def is_valid_plate_text(self, text: str) -> bool:
        """
        Check if text content is consistent with license plate patterns
        
        Args:
            text: Detected text string
            
        Returns:
            Boolean indicating if text looks like a license plate
        """
        # Remove spaces and convert to uppercase
        clean_text = re.sub(r'\s+', '', text.upper())
        
        # Check length
        if not (self.min_chars <= len(clean_text) <= self.max_chars):
            return False
        
        # Check for alphanumeric content
        if not re.match(r'^[A-Z0-9-]+$', clean_text):
            return False
        
        # Check for reasonable mix of letters and numbers
        letter_count = len(re.findall(r'[A-Z]', clean_text))
        number_count = len(re.findall(r'[0-9]', clean_text))
        
        # Must have at least one letter and one number
        if letter_count == 0 or number_count == 0:
            return False
        
        # Common license plate patterns (basic validation)
        patterns = [
            r'^[A-Z]{1,3}[0-9]{1,4}$',  # ABC123
            r'^[0-9]{1,3}[A-Z]{1,3}$',  # 123ABC
            r'^[A-Z]{1,3}[0-9]{1,3}[A-Z]{1,3}$',  # AB123C
            r'^[A-Z0-9]{4,8}$',  # General alphanumeric
        ]
        
        return any(re.match(pattern, clean_text) for pattern in patterns)
    
    def calculate_plate_score(self, text: str, aspect_ratio: float, confidence: float) -> float:
        """
        Calculate a score for how likely a detection is a license plate
        
        Args:
            text: Detected text
            aspect_ratio: Width/height ratio
            confidence: OCR confidence
            
        Returns:
            Plate likelihood score (0-1)
        """
        score = confidence * 0.4  # Base confidence weight
        
        # Aspect ratio score (prefer 3:1 to 5:1 ratios)
        ideal_ratio = 4.0
        ratio_diff = abs(aspect_ratio - ideal_ratio) / ideal_ratio
        ratio_score = max(0, 1 - ratio_diff) * 0.3
        
        # Text pattern score
        clean_text = re.sub(r'\s+', '', text.upper())
        text_score = 0.3
        
        # Bonus for typical license plate length
        if 5 <= len(clean_text) <= 7:
            text_score += 0.1
        
        # Bonus for good letter/number balance
        letter_count = len(re.findall(r'[A-Z]', clean_text))
        number_count = len(re.findall(r'[0-9]', clean_text))
        total_chars = len(clean_text)
        
        if total_chars > 0:
            letter_ratio = letter_count / total_chars
            if 0.2 <= letter_ratio <= 0.8:  # Good mix
                text_score += 0.1
        
        return min(1.0, score + ratio_score + text_score)
    
    def detect_license_plates(self, image: np.ndarray) -> List[Dict]:
        """
        Main detection pipeline: detect text regions and filter for license plates
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected license plates with bounding boxes and metadata
        """
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Detect text regions
        text_regions = self.detect_text_regions(processed_image)
        
        # Filter for license plates
        license_plates = self.filter_license_plates(text_regions, image.shape[:2])
        
        # Add processing metadata
        total_time = time.time() - start_time
        
        for plate in license_plates:
            plate['total_processing_time'] = total_time
            plate['detection_method'] = 'PaddleOCR + Geometric Filtering'
        
        return license_plates
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve detection accuracy
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV uses BGR, PaddleOCR expects RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is not too large (for performance)
        height, width = image.shape[:2]
        max_dimension = 1920
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    def extract_plate_text(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Extract and recognize text from a specific plate region
        
        Args:
            image: Full image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            Tuple of (recognized_text, confidence)
        """
        try:
            x_min, y_min, x_max, y_max = bbox
            
            # Crop plate region with padding
            padding = 5
            crop_x_min = max(0, x_min - padding)
            crop_y_min = max(0, y_min - padding)
            crop_x_max = min(image.shape[1], x_max + padding)
            crop_y_max = min(image.shape[0], y_max + padding)
            
            plate_crop = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            
            if plate_crop.size == 0:
                return "", 0.0
            
            # Enhanced preprocessing for plate region
            plate_processed = self.enhance_plate_crop(plate_crop)
            
            # OCR on cropped region
            results = self.ocr.ocr(plate_processed, cls=False, rec=True)
            
            if results and results[0]:
                # Combine all text from the plate region
                texts = []
                confidences = []
                
                for line in results[0]:
                    if len(line) >= 2:
                        _, (text, confidence) = line
                        texts.append(text.strip())
                        confidences.append(confidence)
                
                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = sum(confidences) / len(confidences)
                    return combined_text, avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            print(f"Error in plate text extraction: {e}")
            return "", 0.0
    
    def enhance_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements to license plate crop for better OCR
        
        Args:
            plate_crop: Cropped plate region
            
        Returns:
            Enhanced plate image
        """
        if len(plate_crop.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = plate_crop.copy()
        
        # Resize if too small
        height, width = gray.shape
        if height < 32 or width < 100:
            scale = max(32 / height, 100 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced