#!/usr/bin/env python3
"""
Batch process images for eye tracking analysis
Runs iris detection and gaze estimation on a directory of images
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Import our modules
from iris_detect import detect_iris
from gaze_estimator import estimate_gaze
from cv_utils import list_images


def process_directory(directory: str, output_dir: str = None, visualize: bool = True) -> dict:
    """
    Process all images in a directory
    
    Returns summary with all results
    """
    images = list_images(directory)
    
    if not images:
        return {"error": f"No images found in {directory}"}
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "directory": directory,
        "total_images": len(images),
        "processed": 0,
        "failed": 0,
        "eyes_detected": 0,
        "images": []
    }
    
    for img_path in images:
        img_result = {
            "file": img_path.name,
            "status": "pending"
        }
        
        try:
            # Run iris detection
            out_path = None
            if output_dir:
                out_path = str(Path(output_dir) / img_path.name)
            
            iris_result = detect_iris(str(img_path), out_path)
            
            if "error" in iris_result:
                img_result["iris_status"] = "failed"
                img_result["iris_error"] = iris_result["error"]
                results["failed"] += 1
            else:
                img_result["iris_status"] = "success"
                img_result["iris_data"] = {
                    "left_iris": iris_result.get("left_iris_center"),
                    "right_iris": iris_result.get("right_iris_center")
                }
                results["eyes_detected"] += 1
                
                # Run gaze estimation
                gaze_out = None
                if output_dir and visualize:
                    gaze_out = str(Path(output_dir) / f"gaze_{img_path.name}")
                
                gaze_result = estimate_gaze(str(img_path), gaze_out)
                
                if "error" not in gaze_result:
                    img_result["gaze"] = gaze_result.get("gaze")
                
                results["processed"] += 1
                
        except Exception as e:
            img_result["status"] = "error"
            img_result["error"] = str(e)
            results["failed"] += 1
        
        results["images"].append(img_result)
    
    # Summary stats
    results["summary"] = {
        "processed": results["processed"],
        "failed": results["failed"],
        "eyes_detected": results["eyes_detected"],
        "detection_rate": round(results["eyes_detected"] / results["total_images"] * 100, 1) if results["total_images"] > 0 else 0
    }
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 batch_process.py <directory> [output_dir]")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Processing {directory}...")
    results = process_directory(directory, output_dir)
    
    print(json.dumps(results, indent=2))
    
    # Save results
    results_file = "batch_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
