#!/usr/bin/env python3
"""
License Plate Blur - Main Application Entry Point
Provides command-line interface and application launcher
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

def run_streamlit_app():
    """Launch the Streamlit web interface"""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path),
        "--server.port", "8000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print("🚀 Starting License Plate Blur Streamlit App...")
    print(f"📱 Access the app at: http://localhost:8000")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down License Plate Blur app...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start app: {e}")
        sys.exit(1)

def run_api_server():
    """Launch the FastAPI backend server"""
    try:
        import uvicorn
        from .api import app
        
        print("🚀 Starting License Plate Blur API Server...")
        print(f"🔌 API available at: http://localhost:8001")
        print(f"📚 API docs at: http://localhost:8001/docs")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            reload=False,
            access_log=True
        )
        
    except ImportError:
        print("❌ FastAPI/Uvicorn not available. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

def run_demo():
    """Run a quick demonstration of the detection capabilities"""
    try:
        import cv2
        import numpy as np
        from .detector import LicensePlateDetector
        from .privacy import PrivacyProtector, BlurMode
        
        print("🔍 Running License Plate Detection Demo...")
        
        # Initialize components
        detector = LicensePlateDetector()
        privacy_protector = PrivacyProtector()
        
        # Create a simple test image with text
        test_image = np.ones((300, 600, 3), dtype=np.uint8) * 255
        
        # Add some text that might look like license plates
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "ABC-1234", (100, 150), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "XYZ-5678", (350, 200), font, 1, (0, 0, 0), 2)
        
        print("🎯 Detecting license plates...")
        detections = detector.detect_license_plates(test_image)
        
        print(f"✅ Found {len(detections)} potential license plates:")
        for i, detection in enumerate(detections):
            print(f"  Plate {i+1}: {detection.get('text', 'N/A')} (confidence: {detection['confidence']:.2f})")
        
        if detections:
            print("🔒 Applying privacy protection...")
            bboxes = [d["bbox"] for d in detections]
            blurred_image = privacy_protector.blur_multiple_regions(
                test_image, bboxes, BlurMode.GAUSSIAN, 15
            )
            print("✅ Privacy protection applied successfully!")
        
        print("🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)

def show_system_info():
    """Display system information and dependencies"""
    print("🔍 License Plate Blur - System Information")
    print("=" * 50)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Check dependencies
    dependencies = [
        ("OpenCV", "cv2"),
        ("PaddleOCR", "paddleocr"),
        ("NumPy", "numpy"),
        ("Streamlit", "streamlit"),
        ("FastAPI", "fastapi"),
        ("MoviePy", "moviepy"),
        ("FFmpeg Python", "ffmpeg")
    ]
    
    print("\nDependencies:")
    for name, module in dependencies:
        try:
            __import__(module)
            version = getattr(__import__(module), "__version__", "unknown")
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: Not installed")
    
    # System info
    print(f"\nSystem: {os.name}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Application Path: {Path(__file__).parent}")

def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="License Plate Blur - ALPR Privacy Protection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py streamlit    # Launch web interface
  python main.py api         # Launch API server
  python main.py demo        # Run detection demo
  python main.py info        # Show system information
        """
    )
    
    parser.add_argument(
        "command",
        choices=["streamlit", "api", "demo", "info"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="License Plate Blur v1.0.0"
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    print("🔒 License Plate Blur - ALPR Privacy Protection")
    print("Advanced AI-powered license plate detection and blurring")
    print("=" * 60)
    
    if args.command == "streamlit":
        run_streamlit_app()
    elif args.command == "api":
        run_api_server()
    elif args.command == "demo":
        run_demo()
    elif args.command == "info":
        show_system_info()

if __name__ == "__main__":
    main()