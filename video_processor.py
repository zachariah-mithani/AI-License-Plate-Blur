"""
Video Processing Module
Handles video frame extraction, processing, and reconstruction
"""

import cv2
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple
import os
import tempfile
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None

try:
    from .detector import LicensePlateDetector
    from .privacy import PrivacyProtector, BlurMode
except ImportError:
    from detector import LicensePlateDetector
    from privacy import PrivacyProtector, BlurMode

class VideoProcessor:
    def __init__(self, detector: LicensePlateDetector, privacy_protector: PrivacyProtector):
        """
        Initialize video processor
        
        Args:
            detector: License plate detector instance
            privacy_protector: Privacy protection instance
        """
        self.detector = detector
        self.privacy_protector = privacy_protector
        
    def process_video(self, input_path: str, output_path: str, 
                     blur_mode: BlurMode = BlurMode.GAUSSIAN,
                     blur_intensity: int = 15,
                     progress_callback: Optional[Callable[[float], None]] = None) -> Dict:
        """
        Process video to blur license plates
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            blur_mode: Blurring method to use
            blur_intensity: Intensity of blurring
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results and statistics
        """
        try:
            # Get video information
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set up video writer
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                raise ValueError(f"Could not create output video: {output_path}")
            
            # Processing statistics
            stats = {
                "total_frames": total_frames,
                "processed_frames": 0,
                "frames_with_plates": 0,
                "total_plates_detected": 0,
                "detections_by_frame": [],
                "processing_time": 0,
                "video_info": {
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration": total_frames / fps if fps > 0 else 0
                }
            }
            
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_results = self.process_frame(frame, blur_mode, blur_intensity)
                processed_frame = frame_results["processed_frame"]
                detections = frame_results["detections"]
                
                # Write processed frame
                out.write(processed_frame)
                
                # Update statistics
                stats["processed_frames"] += 1
                if detections:
                    stats["frames_with_plates"] += 1
                    stats["total_plates_detected"] += len(detections)
                    stats["detections_by_frame"].append({
                        "frame": frame_number,
                        "timestamp": frame_number / fps if fps > 0 else 0,
                        "detections": len(detections),
                        "plates": [{"text": d.get("text", ""), "confidence": d.get("confidence", 0)} for d in detections]
                    })
                
                # Update progress
                if progress_callback:
                    progress = (frame_number + 1) / total_frames
                    progress_callback(progress)
                
                frame_number += 1
            
            # Clean up
            cap.release()
            out.release()
            
            # Add audio back if present (using moviepy for simplicity)
            self._add_audio_to_video(input_path, output_path)
            
            return {
                "success": True,
                "statistics": stats,
                "output_path": output_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "statistics": {}
            }
    
    def process_frame(self, frame: np.ndarray, blur_mode: BlurMode, blur_intensity: int) -> Dict:
        """
        Process a single video frame
        
        Args:
            frame: Input frame
            blur_mode: Blurring method
            blur_intensity: Blur intensity
            
        Returns:
            Dictionary with processed frame and detection results
        """
        # Detect license plates in frame
        detections = self.detector.detect_license_plates(frame)
        
        # Apply blurring to detected plates
        processed_frame = frame.copy()
        bboxes = [d["bbox"] for d in detections]
        
        if bboxes:
            processed_frame = self.privacy_protector.blur_multiple_regions(
                processed_frame, bboxes, blur_mode, blur_intensity
            )
        
        return {
            "processed_frame": processed_frame,
            "detections": detections,
            "frame_shape": frame.shape
        }
    
    def extract_frames(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract sample frames from video for preview
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of extracted frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame intervals
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            frame_indices = [i * step for i in range(max_frames)]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def create_video_preview(self, input_path: str, output_path: str, 
                           blur_mode: BlurMode = BlurMode.GAUSSIAN,
                           blur_intensity: int = 15,
                           max_frames: int = 30) -> Dict:
        """
        Create a short preview of video processing
        
        Args:
            input_path: Input video path
            output_path: Output preview path
            blur_mode: Blurring method
            blur_intensity: Blur intensity
            max_frames: Maximum frames for preview
            
        Returns:
            Preview creation results
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Set up output
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process limited frames
            frames_to_process = min(max_frames, total_frames)
            step = max(1, total_frames // frames_to_process)
            
            detections_summary = []
            
            for i in range(0, total_frames, step):
                if len(detections_summary) >= frames_to_process:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Process frame
                frame_results = self.process_frame(frame, blur_mode, blur_intensity)
                out.write(frame_results["processed_frame"])
                
                # Collect detection info
                if frame_results["detections"]:
                    detections_summary.extend(frame_results["detections"])
            
            cap.release()
            out.release()
            
            return {
                "success": True,
                "preview_path": output_path,
                "frames_processed": frames_to_process,
                "detections_found": len(detections_summary),
                "sample_detections": detections_summary[:5]  # First 5 detections
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _add_audio_to_video(self, input_path: str, output_path: str):
        """
        Add audio from input video to processed output video
        
        Args:
            input_path: Original video with audio
            output_path: Processed video without audio
        """
        try:
            if not MOVIEPY_AVAILABLE or not FFMPEG_AVAILABLE:
                print("Warning: MoviePy or FFmpeg not available, skipping audio preservation")
                return
                
            # Check if input has audio
            with VideoFileClip(input_path) as video:
                if video.audio is not None:
                    # Create temporary file for video with audio
                    temp_path = output_path + "_temp.mp4"
                    
                    # Use ffmpeg to combine processed video with original audio
                    input_video = ffmpeg.input(output_path)
                    input_audio = ffmpeg.input(input_path)
                    
                    out = ffmpeg.output(
                        input_video['v'], 
                        input_audio['a'], 
                        temp_path,
                        vcodec='copy', 
                        acodec='aac',
                        strict='experimental'
                    )
                    
                    ffmpeg.run(out, overwrite_output=True, quiet=True)
                    
                    # Replace original output with version that has audio
                    os.replace(temp_path, output_path)
                    
        except Exception as e:
            print(f"Warning: Could not add audio to video: {e}")
            # Continue without audio - not a critical error
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get detailed information about a video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            info = {
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
                "duration": 0
            }
            
            if info["fps"] > 0:
                info["duration"] = info["frame_count"] / info["fps"]
            
            cap.release()
            
            # Try to get file size
            if os.path.exists(video_path):
                info["file_size"] = os.path.getsize(video_path)
            
            return info
            
        except Exception as e:
            return {"error": str(e)}
    
    def validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate if video file is processable
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(video_path):
            return False, "Video file does not exist"
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file - may be corrupted or unsupported format"
            
            # Try to read first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False, "Cannot read frames from video"
            
            # Check basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width <= 0 or height <= 0:
                cap.release()
                return False, "Invalid video dimensions"
            
            # Check file size (limit to reasonable size for processing)
            file_size = os.path.getsize(video_path)
            max_size = 500 * 1024 * 1024  # 500MB limit
            
            if file_size > max_size:
                cap.release()
                return False, f"Video file too large (max {max_size // (1024*1024)}MB)"
            
            cap.release()
            return True, "Video file is valid"
            
        except Exception as e:
            return False, f"Error validating video: {str(e)}"