#!/usr/bin/env python3
"""
Gaze estimation using iris position
Estimates where someone is looking based on iris landmarks relative to eye corners
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Optional


def estimate_gaze(image_path: str, output_path: str = None) -> dict:
    """
    Estimate gaze direction from iris position
    
    Returns normalized gaze vector and screen position estimate
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load {image_path}"}
    
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # MediaPipe setup
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
    
    # Landmark indices
    LEFT_IRIS = [474, 475, 476, 477, 478, 479]
    RIGHT_IRIS = [468, 469, 470, 471, 472, 473]
    LEFT_EYE_CORNERS = [362, 263]
    RIGHT_EYE_CORNERS = [33, 133]
    
    def get_coords(idx: int) -> Tuple[float, float]:
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)
    
    def get_centroid(indices: list) -> Tuple[float, float]:
        points = [get_coords(i) for i in indices]
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
    
    # Get iris centers
    left_iris = get_centroid(LEFT_IRIS)
    right_iris = get_centroid(RIGHT_IRIS)
    
    # Get eye corners (inner and outer)
    left_inner = get_coords(263)
    left_outer = get_coords(362)
    right_inner = get_coords(33)
    right_outer = get_coords(133)
    
    # Calculate normalized iris position within eye [0, 1]
    # 0 = toward inner corner, 1 = toward outer corner
    def normalized_iris_position(iris_pos, inner_corner, outer_corner) -> float:
        eye_width = outer_corner[0] - inner_corner[0]
        if abs(eye_width) < 1e-6:
            return 0.5
        # For left eye: inner is right, outer is left
        # For right eye: inner is left, outer is right
        normalized = (iris_pos[0] - inner_corner[0]) / eye_width
        return max(0, min(1, normalized))
    
    left_norm = normalized_iris_position(left_iris, left_inner, left_outer)
    right_norm = normalized_iris_position(right_iris, right_inner, right_outer)
    
    # Average horizontal position (0 = left, 1 = right)
    horizontal = (left_norm + (1 - right_norm)) / 2
    
    # Vertical position (simplified)
    left_vert = (left_iris[1] - left_inner[1]) / (left_outer[1] - left_inner[1] + 1e-6)
    right_vert = (right_iris[1] - right_inner[1]) / (right_outer[1] - right_inner[1] + 1e-6)
    vertical = (left_vert + right_vert) / 2
    vertical = max(0, min(1, vertical))
    
    # Gaze direction (normalized)
    gaze_x = (horizontal - 0.5) * 2  # -1 to 1
    gaze_y = (0.5 - vertical) * 2    # -1 to 1 (inverted: up is positive)
    
    # Draw visualization
    # Eye boxes
    cv2.rectangle(img, 
                  (int(left_inner[0]) - 20, int(left_inner[1]) - 10),
                  (int(left_outer[0]) + 20, int(left_outer[1]) + 10),
                  (100, 100, 100), 1)
    cv2.rectangle(img,
                  (int(right_inner[0]) - 20, int(right_inner[1]) - 10),
                  (int(right_outer[0]) + 20, int(right_outer[1]) + 10),
                  (100, 100, 100), 1)
    
    # Iris positions
    cv2.circle(img, (int(left_iris[0]), int(left_iris[1])), 8, (0, 255, 255), -1)
    cv2.circle(img, (int(right_iris[0]), int(right_iris[1])), 8, (0, 255, 255), -1)
    
    # Gaze arrow
    center_x = int((left_iris[0] + right_iris[0]) / 2)
    center_y = int((left_iris[1] + right_iris[1]) / 2)
    arrow_end_x = int(center_x + gaze_x * 50)
    arrow_end_y = int(center_y + gaze_y * 50)
    cv2.arrowedLine(img, (center_x, center_y), (arrow_end_x, arrow_end_y), 
                    (255, 0, 0), 2, tipLength=0.3)
    
    if output_path:
        cv2.imwrite(output_path, img)
    else:
        output_path = str(Path(image_path).with_suffix('.gaze.jpg'))
        cv2.imwrite(output_path, img)
    
    return {
        "faces": 1,
        "gaze": {
            "x": round(gaze_x, 3),  # -1 (left) to 1 (right)
            "y": round(gaze_y, 3)   # -1 (down) to 1 (up)
        },
        "iris_position": {
            "left": round(left_norm, 3),
            "right": round(right_norm, 3)
        },
        "horizontal": round(horizontal, 3),
        "vertical": round(vertical, 3),
        "output_saved": output_path
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gaze_estimator.py <image> [output]")
        sys.exit(1)
    
    image = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = estimate_gaze(image, output)
    print(result)
