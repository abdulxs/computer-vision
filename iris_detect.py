#!/usr/bin/env python3
"""
Iris detection using MediaPipe FaceMesh
Extracts iris landmarks for eye tracking applications
"""

import cv2
import mediapipe as mp
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def detect_iris(image_path: str, output_path: str = None) -> dict:
    """
    Detect iris landmarks in an image using MediaPipe FaceMesh
    
    Returns dict with iris center, pupil, and eye corner landmarks
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load {image_path}"}
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    results = face_mesh.process(rgb_img)
    
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}
    
    landmarks = results.multi_face_landmarks[0]
    
    # Iris landmarks (from MediaPipe FaceMesh)
    # Left eye: 468-473, Right eye: 474-479
    LEFT_IRIS = [474, 475, 476, 477, 478, 479]
    RIGHT_IRIS = [468, 469, 470, 471, 472, 473]
    
    # Eye corners
    LEFT_EYE_CORNERS = [362, 263]  # outer, inner
    RIGHT_EYE_CORNERS = [33, 133]   # inner, outer
    
    def get_landmark_coords(landmark_indices: List[int]) -> List[Tuple[float, float]]:
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in landmark_indices]
    
    left_iris = get_landmark_coords(LEFT_IRIS)
    right_iris = get_landmark_coords(RIGHT_IRIS)
    left_corners = get_landmark_coords(LEFT_EYE_CORNERS)
    right_corners = get_landmark_coords(RIGHT_EYE_CORNERS)
    
    # Calculate iris centers
    def centroid(points: List[Tuple[float, float]]) -> Tuple[float, float]:
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return (x, y)
    
    left_center = centroid(left_iris)
    right_center = centroid(right_iris)
    
    # Draw iris circles
    for center in [left_center, right_center]:
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 255), -1)
    
    # Draw eye corners
    for corner in left_corners + right_corners:
        cv2.circle(img, (int(corner[0]), int(corner[1])), 3, (0, 255, 0), -1)
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        output_path = str(Path(image_path).with_suffix('.iris.jpg'))
        cv2.imwrite(output_path, img)
    
    return {
        "faces_detected": 1,
        "left_iris_center": left_center,
        "right_iris_center": right_center,
        "left_eye_corners": left_corners,
        "right_eye_corners": right_corners,
        "output_saved": output_path
    }


def batch_detect(directory: str, output_dir: str = None) -> List[dict]:
    """Process all images in a directory"""
    from cv_utils import list_images
    
    images = list_images(directory)
    results = []
    
    for img_path in images:
        out_path = None
        if output_dir:
            out_path = str(Path(output_dir) / img_path.name)
        
        result = detect_iris(str(img_path), out_path)
        result["file"] = img_path.name
        results.append(result)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 iris_detect.py <image> [output]")
        sys.exit(1)
    
    image = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = detect_iris(image, output)
    print(result)
