#!/usr/bin/env python3
"""
Computer Vision CLI
Unified interface for all eye tracking utilities
"""

import argparse
import sys
from pathlib import Path

from cv_utils import list_images, batch_info
from iris_detect import detect_iris
from gaze_estimator import estimate_gaze
from batch_process import process_directory


def cmd_list(args):
    """List images in directory"""
    images = list_images(args.directory)
    for img in images:
        print(img.name)
    print(f"\n{len(images)} images found")


def cmd_info(args):
    """Get image info"""
    info = batch_info(args.directory)
    import json
    print(json.dumps(info, indent=2))


def cmd_iris(args):
    """Detect iris in image(s)"""
    if args.directory:
        results = process_directory(args.directory, args.output, visualize=False)
        import json
        print(json.dumps(results, indent=2))
    else:
        result = detect_iris(args.image, args.output)
        import json
        print(json.dumps(result, indent=2))


def cmd_gaze(args):
    """Estimate gaze in image(s)"""
    if args.directory:
        # Process directory with gaze estimation
        from iris_detect import detect_iris as detect
        images = list_images(args.directory)
        for img in images:
            out = None
            if args.output:
                out = str(Path(args.output) / f"gaze_{img.name}")
            result = estimate_gaze(str(img), out)
            print(f"{img.name}: {result.get('gaze', result.get('error'))}")
    else:
        result = estimate_gaze(args.image, args.output)
        import json
        print(json.dumps(result, indent=2))


def cmd_batch(args):
    """Full batch processing"""
    results = process_directory(args.directory, args.output, visualize=True)
    import json
    print(json.dumps(results["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Computer Vision CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list
    p_list = subparsers.add_parser("list", help="List images in directory")
    p_list.add_argument("directory", nargs="?", default=".", help="Directory to list")
    p_list.set_defaults(func=cmd_list)
    
    # info
    p_info = subparsers.add_parser("info", help="Get image info")
    p_info.add_argument("directory", nargs="?", default=".", help="Directory to scan")
    p_info.set_defaults(func=cmd_info)
    
    # iris
    p_iris = subparsers.add_parser("iris", help="Detect iris landmarks")
    p_iris.add_argument("image", nargs="?", help="Single image file")
    p_iris.add_argument("-d", "--directory", help="Process directory")
    p_iris.add_argument("-o", "--output", help="Output directory")
    p_iris.set_defaults(func=cmd_iris)
    
    # gaze
    p_gaze = subparsers.add_parser("gaze", help="Estimate gaze direction")
    p_gaze.add_argument("image", nargs="?", help="Single image file")
    p_gaze.add_argument("-d", "--directory", help="Process directory")
    p_gaze.add_argument("-o", "--output", help="Output directory")
    p_gaze.set_defaults(func=cmd_gaze)
    
    # batch
    p_batch = subparsers.add_parser("batch", help="Full batch processing")
    p_batch.add_argument("directory", help="Directory to process")
    p_batch.add_argument("-o", "--output", help="Output directory")
    p_batch.set_defaults(func=cmd_batch)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
