# Computer Vision

Experiments with vision models and image processing.

## Contents

- `detect_and_annotate_images.ipynb` - Colab notebook for image detection/annotation

## Notebook Details

- 238 lines
- Originally from: Colab (google colab link in markdown)
- Run on: Google Colab

## Goals

- [x] Explore the notebook
- [x] Added iris detection utilities (2026-02-16)
- [ ] Add more vision experiments
- [ ] Document findings

## Available Scripts

- `cv_utils.py` - Image listing and info utilities
- `face_detect.py` - OpenCV Haar cascade face detection
- `iris_detect.py` - MediaPipe iris landmark detection
- `gaze_estimator.py` - Gaze direction estimation from iris position
- `batch_process.py` - Batch process directory of images
- `cli.py` - Unified CLI interface

## Usage

```bash
# List images
python cli.py list ./images

# Full batch processing
python cli.py batch ./images -o ./output

# Single image gaze estimation
python cli.py gaze photo.jpg -o result.jpg
```

## Dependencies

```bash
pip install opencv-python mediapipe numpy
```
