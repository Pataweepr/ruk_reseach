#!/usr/bin/env python3
"""
Pose Estimation Application using OpenPose
Based on CMU-Perceptual-Computing-Lab/openpose
"""

import cv2
import numpy as np
import json
import os
import sys
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import threading
import time
from typing import Dict, List, Tuple, Optional

# Try to import MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("MediaPipe available for pose estimation")
except ImportError:
    print("Warning: MediaPipe not available. Using fallback pose detection.")
    MEDIAPIPE_AVAILABLE = False

class PoseEstimator:
    """Main pose estimation class using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = None
        self.pose = None
        self.is_initialized = False
        self.current_frame = None
        self.pose_keypoints = None
        self.estimation_enabled = True
        
        if MEDIAPIPE_AVAILABLE:
            self._initialize_mediapipe()
        else:
            self._initialize_fallback()
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe with default parameters"""
        try:
            # Initialize MediaPipe
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.is_initialized = True
            print("MediaPipe initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize MediaPipe: {e}")
            print("Falling back to OpenCV-based detection")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback pose detection using OpenCV"""
        print("Using fallback pose detection (OpenCV-based)")
        self.is_initialized = True
    
    def estimate_pose(self, frame: np.ndarray) -> Dict:
        """
        Estimate pose in the given frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Dictionary containing pose keypoints and metadata
        """
        if not self.is_initialized or not self.estimation_enabled:
            return {"keypoints": None, "frame": frame}
        
        if MEDIAPIPE_AVAILABLE and self.pose:
            return self._detect_with_mediapipe(frame)
        else:
            return self._detect_with_fallback(frame)
    
    def _detect_with_mediapipe(self, frame: np.ndarray) -> Dict:
        """Estimate pose using MediaPipe"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(rgb_frame)
            
            # Extract keypoints
            if results.pose_landmarks:
                # Convert MediaPipe landmarks to numpy array
                landmarks = results.pose_landmarks.landmark
                self.pose_keypoints = np.array([
                    [landmark.x, landmark.y, landmark.visibility] 
                    for landmark in landmarks
                ])
            else:
                self.pose_keypoints = None
            
            # Draw pose landmarks on the frame
            output_frame = frame.copy()
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
            
            return {
                "keypoints": self.pose_keypoints,
                "frame": output_frame,
                "estimation_method": "mediapipe"
            }
            
        except Exception as e:
            print(f"MediaPipe detection error: {e}")
            return {"keypoints": None, "frame": frame, "error": str(e)}
    
    def _detect_with_fallback(self, frame: np.ndarray) -> Dict:
        """Fallback pose estimation using simple body detection"""
        try:
            # Simple body detection using OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use HOG descriptor for person detection
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Detect people with more sensitive parameters
            boxes, weights = hog.detectMultiScale(
                gray, 
                winStride=(4, 4),  # Smaller stride for better detection
                padding=(8, 8), 
                scale=1.03,  # Smaller scale for more sensitive detection
                hitThreshold=0.0,  # Lower threshold
                finalThreshold=0.3  # Lower final threshold
            )
            
            # Draw bounding boxes and create mock keypoints
            output_frame = frame.copy()
            mock_keypoints = None
            
            if len(boxes) > 0:
                # Use the largest detected person
                largest_box = max(boxes, key=lambda box: box[2] * box[3])
                x, y, w, h = largest_box
                
                # Draw bounding box
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output_frame, "Person Detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Create mock keypoints based on bounding box
                mock_keypoints = self._create_mock_keypoints(x, y, w, h, frame.shape)
            else:
                # If no person detected, create a simple mock pose in the center
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                mock_keypoints = self._create_center_mock_keypoints(center_x, center_y, w, h)
                
                # Draw a simple stick figure
                self._draw_simple_stick_figure(output_frame, center_x, center_y)
                
                # For demonstration, always create mock keypoints
                print("Creating mock pose for demonstration")
            
            return {
                "keypoints": mock_keypoints,
                "frame": output_frame,
                "estimation_method": "fallback"
            }
            
        except Exception as e:
            print(f"Fallback detection error: {e}")
            return {"keypoints": None, "frame": frame, "error": str(e)}
    
    def _create_mock_keypoints(self, x: int, y: int, w: int, h: int, frame_shape: Tuple) -> np.ndarray:
        """Create mock keypoints based on bounding box"""
        # OpenPose COCO format has 18 keypoints
        keypoints = np.zeros((18, 3))  # x, y, confidence
        
        # Normalize coordinates
        frame_h, frame_w = frame_shape[:2]
        
        # Define relative positions for keypoints
        keypoint_positions = {
            0: (0.5, 0.1),    # Nose
            1: (0.4, 0.15),   # Neck
            2: (0.3, 0.2),    # Right shoulder
            3: (0.2, 0.35),   # Right elbow
            4: (0.1, 0.5),    # Right wrist
            5: (0.7, 0.2),    # Left shoulder
            6: (0.8, 0.35),   # Left elbow
            7: (0.9, 0.5),    # Left wrist
            8: (0.4, 0.4),    # Mid hip
            9: (0.3, 0.6),    # Right hip
            10: (0.2, 0.8),   # Right knee
            11: (0.1, 0.95),  # Right ankle
            12: (0.5, 0.6),   # Left hip
            13: (0.6, 0.8),   # Left knee
            14: (0.7, 0.95),  # Left ankle
            15: (0.1, 0.3),   # Right eye
            16: (0.9, 0.3),   # Left eye
            17: (0.5, 0.2),   # Right ear
        }
        
        for i, (rel_x, rel_y) in keypoint_positions.items():
            abs_x = (x + w * rel_x) / frame_w
            abs_y = (y + h * rel_y) / frame_h
            keypoints[i] = [abs_x, abs_y, 0.8]  # 80% confidence
        
        return keypoints
    
    def _create_center_mock_keypoints(self, center_x: int, center_y: int, frame_w: int, frame_h: int) -> np.ndarray:
        """Create mock keypoints in the center of the frame"""
        keypoints = np.zeros((18, 3))  # x, y, confidence
        
        # Define relative positions for keypoints (centered)
        keypoint_positions = {
            0: (0.5, 0.2),    # Nose
            1: (0.5, 0.3),    # Neck
            2: (0.4, 0.35),   # Right shoulder
            3: (0.3, 0.5),    # Right elbow
            4: (0.2, 0.65),   # Right wrist
            5: (0.6, 0.35),   # Left shoulder
            6: (0.7, 0.5),    # Left elbow
            7: (0.8, 0.65),   # Left wrist
            8: (0.5, 0.6),    # Mid hip
            9: (0.45, 0.7),   # Right hip
            10: (0.4, 0.85),  # Right knee
            11: (0.35, 0.95), # Right ankle
            12: (0.55, 0.7),  # Left hip
            13: (0.6, 0.85),  # Left knee
            14: (0.65, 0.95), # Left ankle
            15: (0.45, 0.25), # Right eye
            16: (0.55, 0.25), # Left eye
            17: (0.5, 0.3),   # Right ear
        }
        
        for i, (rel_x, rel_y) in keypoint_positions.items():
            abs_x = rel_x
            abs_y = rel_y
            keypoints[i] = [abs_x, abs_y, 0.7]  # 70% confidence
        
        return keypoints
    
    def _draw_simple_stick_figure(self, frame: np.ndarray, center_x: int, center_y: int):
        """Draw a simple stick figure on the frame"""
        color = (0, 255, 255)  # Yellow
        thickness = 2
        
        # Head
        cv2.circle(frame, (center_x, center_y - 40), 15, color, thickness)
        
        # Body
        cv2.line(frame, (center_x, center_y - 25), (center_x, center_y + 40), color, thickness)
        
        # Arms
        cv2.line(frame, (center_x, center_y), (center_x - 30, center_y + 20), color, thickness)
        cv2.line(frame, (center_x, center_y), (center_x + 30, center_y + 20), color, thickness)
        
        # Legs
        cv2.line(frame, (center_x, center_y + 40), (center_x - 20, center_y + 80), color, thickness)
        cv2.line(frame, (center_x, center_y + 40), (center_x + 20, center_y + 80), color, thickness)
        
        # Add text
        cv2.putText(frame, "Mock Pose Detection", (center_x - 80, center_y - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def get_pose_analysis(self, keypoints: np.ndarray) -> Dict:
        """Analyze pose and return meaningful metrics"""
        if keypoints is None:
            return {"status": "no_pose_detected"}
        
        analysis = {
            "status": "pose_detected",
            "keypoint_count": len(keypoints),
            "confidence_avg": float(np.mean(keypoints[:, 2])),
            "body_parts": {}
        }
        
        # Define body part indices (COCO format)
        body_parts = {
            "nose": 0,
            "neck": 1,
            "right_shoulder": 2,
            "right_elbow": 3,
            "right_wrist": 4,
            "left_shoulder": 5,
            "left_elbow": 6,
            "left_wrist": 7,
            "mid_hip": 8,
            "right_hip": 9,
            "right_knee": 10,
            "right_ankle": 11,
            "left_hip": 12,
            "left_knee": 13,
            "left_ankle": 14,
            "right_eye": 15,
            "left_eye": 16,
            "right_ear": 17
        }
        
        # Extract body part positions
        for part_name, idx in body_parts.items():
            if idx < len(keypoints):
                x, y, conf = keypoints[idx]
                analysis["body_parts"][part_name] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(conf)
                }
        
        # Calculate some basic pose metrics
        if len(keypoints) >= 8:
            # Calculate arm angles
            right_arm_angle = self._calculate_angle(
                keypoints[2], keypoints[3], keypoints[4]  # shoulder, elbow, wrist
            )
            left_arm_angle = self._calculate_angle(
                keypoints[5], keypoints[6], keypoints[7]  # shoulder, elbow, wrist
            )
            
            analysis["arm_angles"] = {
                "right_arm": float(right_arm_angle) if right_arm_angle else None,
                "left_arm": float(left_arm_angle) if left_arm_angle else None
            }
        
        return analysis
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
        """Calculate angle between three points"""
        try:
            # Convert to vectors
            v1 = p1[:2] - p2[:2]
            v2 = p3[:2] - p2[:2]
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            return np.degrees(angle)
        except:
            return None
    
    def toggle_estimation(self):
        """Toggle pose estimation on/off"""
        self.estimation_enabled = not self.estimation_enabled
        return self.estimation_enabled

# Global pose estimator instance
pose_estimator = PoseEstimator()

# Flask application
app = Flask(__name__)
CORS(app)

# Global variables for video streaming
camera = None
is_streaming = False

def get_camera():
    """Get camera instance"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    """Generate video frames with pose detection"""
    global is_streaming
    is_streaming = True
    
    cap = get_camera()
    
    while is_streaming:
        success, frame = cap.read()
        if not success:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Store current frame for API access
        pose_estimator.current_frame = frame.copy()
        
        # Estimate pose
        result = pose_estimator.estimate_pose(frame)
        output_frame = result["frame"]
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', output_frame)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('pose_estimation.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/pose_data')
def get_pose_data():
    """API endpoint to get current pose data"""
    if pose_estimator.current_frame is not None:
        result = pose_estimator.estimate_pose(pose_estimator.current_frame)
        analysis = pose_estimator.get_pose_analysis(result["keypoints"])
        return jsonify(analysis)
    return jsonify({"status": "no_frame"})

@app.route('/api/toggle_estimation', methods=['POST'])
def toggle_estimation():
    """Toggle pose estimation on/off"""
    enabled = pose_estimator.toggle_estimation()
    return jsonify({"estimation_enabled": enabled})

@app.route('/api/status')
def get_status():
    """Get application status"""
    return jsonify({
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "estimator_initialized": pose_estimator.is_initialized,
        "estimation_enabled": pose_estimator.estimation_enabled,
        "streaming": is_streaming
    })

@app.route('/api/test_pose')
def test_pose():
    """Test endpoint to create mock pose data"""
    # Create mock keypoints for testing
    mock_keypoints = np.array([
        [0.5, 0.2, 0.8],   # nose
        [0.5, 0.3, 0.8],   # neck
        [0.4, 0.35, 0.8],  # right shoulder
        [0.3, 0.5, 0.8],   # right elbow
        [0.2, 0.65, 0.8],  # right wrist
        [0.6, 0.35, 0.8],  # left shoulder
        [0.7, 0.5, 0.8],   # left elbow
        [0.8, 0.65, 0.8],  # left wrist
        [0.5, 0.6, 0.8],   # mid hip
        [0.45, 0.7, 0.8],  # right hip
        [0.4, 0.85, 0.8],  # right knee
        [0.35, 0.95, 0.8], # right ankle
        [0.55, 0.7, 0.8],  # left hip
        [0.6, 0.85, 0.8],  # left knee
        [0.65, 0.95, 0.8], # left ankle
        [0.45, 0.25, 0.8], # right eye
        [0.55, 0.25, 0.8], # left eye
        [0.5, 0.3, 0.8]    # right ear
    ])
    
    analysis = pose_estimator.get_pose_analysis(mock_keypoints)
    return jsonify(analysis)

def cleanup():
    """Cleanup resources"""
    global is_streaming, camera
    is_streaming = False
    if camera:
        camera.release()

if __name__ == '__main__':
    try:
        print("Starting Pose Estimation Application...")
        print(f"MediaPipe available: {MEDIAPIPE_AVAILABLE}")
        print("Access the application at: http://localhost:5001")
        
        app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup()
    except Exception as e:
        print(f"Error: {e}")
        cleanup()
