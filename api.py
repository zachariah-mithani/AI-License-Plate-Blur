"""
FastAPI Backend for License Plate Blur Service
Provides REST API endpoints for detection and blurring operations
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import os
import tempfile
import json
import uuid
from datetime import datetime
import cv2
import numpy as np

try:
    from .detector import LicensePlateDetector
    from .privacy import PrivacyProtector, BlurMode
    from .video_processor import VideoProcessor
except ImportError:
    from detector import LicensePlateDetector
    from privacy import PrivacyProtector, BlurMode
    from video_processor import VideoProcessor

# Pydantic models for API requests/responses
class DetectionRequest(BaseModel):
    confidence_threshold: float = 0.7
    enable_ocr: bool = True

class BlurRequest(BaseModel):
    confidence_threshold: float = 0.7
    blur_mode: str = "gaussian"  # gaussian, pixelate, box
    blur_intensity: int = 15
    enable_ocr: bool = True

class DetectionResult(BaseModel):
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    text: str
    confidence: float
    plate_score: float

class ProcessingResult(BaseModel):
    success: bool
    processing_time: float
    detections: List[DetectionResult]
    total_plates: int
    media_type: str
    
class BlurResult(BaseModel):
    success: bool
    processing_time: float
    detections: List[DetectionResult]
    total_plates_blurred: int
    media_type: str
    output_file_id: str
    privacy_report: Dict

# Initialize FastAPI app
app = FastAPI(
    title="License Plate Blur API",
    description="Privacy protection service for automatic license plate detection and blurring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
detector = LicensePlateDetector()
privacy_protector = PrivacyProtector()
video_processor = VideoProcessor(detector, privacy_protector)

# Storage for temporary files
TEMP_DIR = tempfile.mkdtemp()
processed_files = {}  # Store processed files with UUIDs

@app.get("/healthz")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "service": "License Plate Blur API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=ProcessingResult)
async def detect_plates(
    file: UploadFile = File(...),
    request: DetectionRequest = DetectionRequest()
):
    """
    Detect license plates in uploaded image or video
    
    Args:
        file: Uploaded image or video file
        request: Detection parameters
        
    Returns:
        Detection results with bounding boxes and text
    """
    try:
        # Validate file type
        if not file.content_type:
            raise HTTPException(status_code=400, detail="Could not determine file type")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.content_type.startswith('image/'):
            result = await _detect_in_image(content, request)
        elif file.content_type.startswith('video/'):
            result = await _detect_in_video(content, request, file.filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/blur", response_model=BlurResult)
async def blur_plates(
    file: UploadFile = File(...),
    request: BlurRequest = BlurRequest()
):
    """
    Detect and blur license plates in uploaded media
    
    Args:
        file: Uploaded image or video file
        request: Blurring parameters
        
    Returns:
        Blurring results with download link
    """
    try:
        # Validate file type
        if not file.content_type:
            raise HTTPException(status_code=400, detail="Could not determine file type")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        if file.content_type.startswith('image/'):
            result = await _blur_image(content, request)
        elif file.content_type.startswith('video/'):
            result = await _blur_video(content, request, file.filename)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blurring failed: {str(e)}")

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download processed file by ID
    
    Args:
        file_id: UUID of processed file
        
    Returns:
        File download response
    """
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = processed_files[file_id]
    file_path = file_info["path"]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer available")
    
    return FileResponse(
        path=file_path,
        filename=file_info["original_name"],
        media_type=file_info["media_type"]
    )

@app.get("/files/{file_id}/info")
async def get_file_info(file_id: str):
    """Get information about a processed file"""
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    return processed_files[file_id]

@app.delete("/files/{file_id}")
async def delete_file(file_id: str):
    """Delete a processed file"""
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = processed_files[file_id]
    file_path = file_info["path"]
    
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        del processed_files[file_id]
        return {"success": True, "message": "File deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

# Helper functions
async def _detect_in_image(content: bytes, request: DetectionRequest) -> ProcessingResult:
    """Detect license plates in image"""
    import time
    
    start_time = time.time()
    
    # Convert bytes to image
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Update detector confidence threshold
    detector.confidence_threshold = request.confidence_threshold
    
    # Detect plates
    detections = detector.detect_license_plates(image)
    
    # Convert to API format
    api_detections = []
    for detection in detections:
        api_detection = DetectionResult(
            bbox=list(detection["bbox"]),
            text=detection.get("text", ""),
            confidence=detection["confidence"],
            plate_score=detection.get("plate_score", 0.0)
        )
        api_detections.append(api_detection)
    
    processing_time = time.time() - start_time
    
    return ProcessingResult(
        success=True,
        processing_time=processing_time,
        detections=api_detections,
        total_plates=len(api_detections),
        media_type="image"
    )

async def _detect_in_video(content: bytes, request: DetectionRequest, filename: str) -> ProcessingResult:
    """Detect license plates in video"""
    import time
    
    start_time = time.time()
    
    # Save video to temporary file
    temp_input = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.mp4")
    
    with open(temp_input, "wb") as f:
        f.write(content)
    
    try:
        # Update detector confidence threshold
        detector.confidence_threshold = request.confidence_threshold
        
        # Extract sample frames and detect plates
        frames = video_processor.extract_frames(temp_input, max_frames=10)
        all_detections = []
        
        for frame in frames:
            frame_detections = detector.detect_license_plates(frame)
            all_detections.extend(frame_detections)
        
        # Convert to API format
        api_detections = []
        for detection in all_detections:
            api_detection = DetectionResult(
                bbox=list(detection["bbox"]),
                text=detection.get("text", ""),
                confidence=detection["confidence"],
                plate_score=detection.get("plate_score", 0.0)
            )
            api_detections.append(api_detection)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            success=True,
            processing_time=processing_time,
            detections=api_detections,
            total_plates=len(api_detections),
            media_type="video"
        )
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_input):
            os.remove(temp_input)

async def _blur_image(content: bytes, request: BlurRequest) -> BlurResult:
    """Blur license plates in image"""
    import time
    
    start_time = time.time()
    
    # Convert bytes to image
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Update detector confidence threshold
    detector.confidence_threshold = request.confidence_threshold
    
    # Detect plates
    detections = detector.detect_license_plates(image)
    
    # Apply blurring
    blur_mode = BlurMode(request.blur_mode)
    bboxes = [d["bbox"] for d in detections]
    
    blurred_image = privacy_protector.blur_multiple_regions(
        image, bboxes, blur_mode, request.blur_intensity
    )
    
    # Save blurred image
    file_id = str(uuid.uuid4())
    output_path = os.path.join(TEMP_DIR, f"blurred_{file_id}.jpg")
    
    cv2.imwrite(output_path, blurred_image)
    
    # Store file info
    processed_files[file_id] = {
        "path": output_path,
        "original_name": f"blurred_image_{file_id}.jpg",
        "media_type": "image/jpeg",
        "created_at": datetime.now().isoformat(),
        "processing_settings": request.dict()
    }
    
    # Generate privacy report
    privacy_report = privacy_protector.generate_privacy_report(
        detections, 
        {"mode": request.blur_mode, "intensity": request.blur_intensity}
    )
    
    # Convert detections to API format
    api_detections = []
    for detection in detections:
        api_detection = DetectionResult(
            bbox=list(detection["bbox"]),
            text=detection.get("text", ""),
            confidence=detection["confidence"],
            plate_score=detection.get("plate_score", 0.0)
        )
        api_detections.append(api_detection)
    
    processing_time = time.time() - start_time
    
    return BlurResult(
        success=True,
        processing_time=processing_time,
        detections=api_detections,
        total_plates_blurred=len(api_detections),
        media_type="image",
        output_file_id=file_id,
        privacy_report=privacy_report
    )

async def _blur_video(content: bytes, request: BlurRequest, filename: str) -> BlurResult:
    """Blur license plates in video"""
    import time
    
    start_time = time.time()
    
    # Save video to temporary file
    temp_input = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.mp4")
    
    with open(temp_input, "wb") as f:
        f.write(content)
    
    try:
        # Update detector confidence threshold
        detector.confidence_threshold = request.confidence_threshold
        
        # Process video
        file_id = str(uuid.uuid4())
        output_path = os.path.join(TEMP_DIR, f"blurred_{file_id}.mp4")
        
        blur_mode = BlurMode(request.blur_mode)
        
        result = video_processor.process_video(
            temp_input, 
            output_path, 
            blur_mode, 
            request.blur_intensity
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Video processing failed"))
        
        # Store file info
        processed_files[file_id] = {
            "path": output_path,
            "original_name": f"blurred_{filename}",
            "media_type": "video/mp4",
            "created_at": datetime.now().isoformat(),
            "processing_settings": request.dict(),
            "video_stats": result["statistics"]
        }
        
        # Generate privacy report
        stats = result["statistics"]
        mock_detections = []  # Video processing returns frame-level stats
        
        privacy_report = privacy_protector.generate_privacy_report(
            mock_detections,
            {"mode": request.blur_mode, "intensity": request.blur_intensity}
        )
        privacy_report.update({
            "video_statistics": stats,
            "frames_processed": stats.get("processed_frames", 0),
            "frames_with_plates": stats.get("frames_with_plates", 0)
        })
        
        processing_time = time.time() - start_time
        
        return BlurResult(
            success=True,
            processing_time=processing_time,
            detections=[],  # Video detections are frame-level
            total_plates_blurred=stats.get("total_plates_detected", 0),
            media_type="video",
            output_file_id=file_id,
            privacy_report=privacy_report
        )
        
    finally:
        # Clean up temp input file
        if os.path.exists(temp_input):
            os.remove(temp_input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)