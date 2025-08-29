"""
Privacy Protection Module
Handles blurring and pixelation of detected license plates
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum

class BlurMode(Enum):
    GAUSSIAN = "gaussian"
    PIXELATE = "pixelate"
    BOX = "box"

class PrivacyProtector:
    def __init__(self):
        """Initialize privacy protection system"""
        pass
    
    def blur_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                   blur_mode: BlurMode = BlurMode.GAUSSIAN, 
                   intensity: int = 15) -> np.ndarray:
        """
        Apply blurring to a specific region of the image
        
        Args:
            image: Input image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            blur_mode: Type of blurring to apply
            intensity: Blur intensity (higher = more blur)
            
        Returns:
            Image with blurred region
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x_min = max(0, min(x_min, width - 1))
        y_min = max(0, min(y_min, height - 1))
        x_max = max(x_min + 1, min(x_max, width))
        y_max = max(y_min + 1, min(y_max, height))
        
        # Create a copy of the image
        result = image.copy()
        
        # Extract the region to blur
        region = result[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return result
        
        # Apply the selected blur method
        if blur_mode == BlurMode.GAUSSIAN:
            blurred_region = self._apply_gaussian_blur(region, intensity)
        elif blur_mode == BlurMode.PIXELATE:
            blurred_region = self._apply_pixelation(region, intensity)
        elif blur_mode == BlurMode.BOX:
            blurred_region = self._apply_box_blur(region, intensity)
        else:
            blurred_region = self._apply_gaussian_blur(region, intensity)
        
        # Replace the region in the result image
        result[y_min:y_max, x_min:x_max] = blurred_region
        
        return result
    
    def _apply_gaussian_blur(self, region: np.ndarray, intensity: int) -> np.ndarray:
        """
        Apply Gaussian blur to image region
        
        Args:
            region: Image region to blur
            intensity: Blur kernel size
            
        Returns:
            Blurred region
        """
        # Ensure kernel size is odd and positive
        kernel_size = max(3, intensity | 1)  # Make odd
        
        return cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)
    
    def _apply_pixelation(self, region: np.ndarray, intensity: int) -> np.ndarray:
        """
        Apply pixelation effect to image region
        
        Args:
            region: Image region to pixelate
            intensity: Pixelation block size
            
        Returns:
            Pixelated region
        """
        height, width = region.shape[:2]
        
        # Calculate downscale factor based on intensity
        block_size = max(2, intensity // 2)
        
        # Calculate new dimensions
        new_width = max(1, width // block_size)
        new_height = max(1, height // block_size)
        
        # Downscale
        small = cv2.resize(region, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Upscale back using nearest neighbor for pixelated effect
        pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return pixelated
    
    def _apply_box_blur(self, region: np.ndarray, intensity: int) -> np.ndarray:
        """
        Apply box filter blur to image region
        
        Args:
            region: Image region to blur
            intensity: Box filter size
            
        Returns:
            Blurred region
        """
        kernel_size = max(3, intensity | 1)  # Make odd
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        return cv2.filter2D(region, -1, kernel)
    
    def blur_multiple_regions(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]],
                             blur_mode: BlurMode = BlurMode.GAUSSIAN, 
                             intensity: int = 15) -> np.ndarray:
        """
        Apply blurring to multiple regions in the image
        
        Args:
            image: Input image
            bboxes: List of bounding boxes
            blur_mode: Type of blurring to apply
            intensity: Blur intensity
            
        Returns:
            Image with all regions blurred
        """
        result = image.copy()
        
        for bbox in bboxes:
            result = self.blur_region(result, bbox, blur_mode, intensity)
        
        return result
    
    def create_comparison_view(self, original: np.ndarray, blurred: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of original and blurred images
        
        Args:
            original: Original image
            blurred: Image with blurred regions
            
        Returns:
            Side-by-side comparison image
        """
        height = max(original.shape[0], blurred.shape[0])
        
        # Resize images to same height if needed
        if original.shape[0] != height:
            original = cv2.resize(original, (int(original.shape[1] * height / original.shape[0]), height))
        if blurred.shape[0] != height:
            blurred = cv2.resize(blurred, (int(blurred.shape[1] * height / blurred.shape[0]), height))
        
        # Create side-by-side comparison
        comparison = np.hstack([original, blurred])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = height / 500  # Scale font with image size
        thickness = max(1, int(height / 300))
        
        cv2.putText(comparison, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(comparison, "Privacy Protected", (original.shape[1] + 10, 30), font, font_scale, (255, 255, 255), thickness)
        
        return comparison
    
    def validate_blur_effectiveness(self, original: np.ndarray, blurred: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> float:
        """
        Validate that blurring has been effectively applied to a region
        
        Args:
            original: Original image
            blurred: Blurred image
            bbox: Bounding box of the blurred region
            
        Returns:
            Effectiveness score (0-1, higher means more effective blurring)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Extract regions
        original_region = original[y_min:y_max, x_min:x_max]
        blurred_region = blurred[y_min:y_max, x_min:x_max]
        
        if original_region.size == 0 or blurred_region.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(original_region.shape) == 3:
            original_gray = cv2.cvtColor(original_region, cv2.COLOR_BGR2GRAY)
            blurred_gray = cv2.cvtColor(blurred_region, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_region
            blurred_gray = blurred_region
        
        # Calculate mean absolute difference
        diff = np.mean(np.abs(original_gray.astype(float) - blurred_gray.astype(float)))
        
        # Calculate variance reduction (blurred should have lower variance)
        original_var = np.var(original_gray)
        blurred_var = np.var(blurred_gray)
        
        if original_var == 0:
            variance_reduction = 1.0
        else:
            variance_reduction = max(0, (original_var - blurred_var) / original_var)
        
        # Combine metrics (normalize diff to 0-1 range)
        diff_score = min(1.0, float(diff) / 50.0)  # Assume max meaningful diff is 50
        
        effectiveness = (diff_score * 0.6) + (float(variance_reduction) * 0.4)
        
        return min(1.0, float(effectiveness))
    
    def add_blur_indicators(self, image: np.ndarray, bboxes: List[Tuple[int, int, int, int]], 
                           confidences: Optional[List[float]] = None) -> np.ndarray:
        """
        Add visual indicators showing where blurring has been applied
        
        Args:
            image: Image with blurred regions
            bboxes: Bounding boxes of blurred regions
            confidences: Optional confidence scores for each detection
            
        Returns:
            Image with blur indicators added
        """
        result = image.copy()
        
        for i, bbox in enumerate(bboxes):
            x_min, y_min, x_max, y_max = bbox
            
            # Draw bounding box
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add label
            label = "BLURRED"
            if confidences and i < len(confidences):
                label += f" ({confidences[i]:.1%})"
            
            # Calculate label position
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = y_min - 5 if y_min - 5 > label_size[1] else y_min + label_size[1] + 5
            
            # Draw label background
            cv2.rectangle(result, (x_min, label_y - label_size[1] - 2), 
                         (x_min + label_size[0] + 4, label_y + 2), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(result, label, (x_min + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return result
    
    def generate_privacy_report(self, detections: List[Dict], 
                               blur_settings: Dict) -> Dict:
        """
        Generate a privacy compliance report
        
        Args:
            detections: List of license plate detections
            blur_settings: Settings used for blurring
            
        Returns:
            Privacy compliance report
        """
        report = {
            "timestamp": np.datetime64('now').astype(str),
            "total_plates_detected": len(detections),
            "total_plates_blurred": len(detections),  # Always blur all detected plates
            "blur_method": blur_settings.get("mode", "gaussian"),
            "blur_intensity": blur_settings.get("intensity", 15),
            "privacy_compliance_level": "FULL",
            "detections": []
        }
        
        for i, detection in enumerate(detections):
            detection_report = {
                "detection_id": i + 1,
                "plate_text": detection.get("text", "N/A"),
                "confidence": detection.get("confidence", 0.0),
                "bounding_box": detection.get("bbox", [0, 0, 0, 0]),
                "blurred": True,
                "privacy_protected": True
            }
            report["detections"].append(detection_report)
        
        return report