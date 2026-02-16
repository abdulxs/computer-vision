#!/usr/bin/env python3
"""
Simple face detection using OpenCV
"""

import cv2
import sys
from pathlib import Path


def detect_faces(image_path: str, output_path: str = None):
    """Detect faces in an image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save output
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")
    else:
        output_path = str(Path(image_path).with_suffix('.faces.jpg'))
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")
    
    print(f"Faces detected: {len(faces)}")
    return len(faces)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 face_detect.py <image> [output]")
        sys.exit(1)
    
    image = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else None
    detect_faces(image, output)
