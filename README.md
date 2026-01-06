# Backend Sign Processing System

This project implements a high-performance computer vision pipeline for detecting signs, stitching panoramic views, and correcting occlusions using multi-view geometry.

# Architecture

1. Filtering: Checks input folders (face1...face4).

2. Stitching: If a folder has multiple images, it uses cv2.Stitcher (Scan mode) to create a single high-res panorama.

3. Detection: A placeholder MockSignDetector identifies signs (segmentation masks).

4. Occlusion Recovery:

  - If a mask is fragmented (indicating occlusion), the system loads auxiliary images from signface folders.

  - It uses ORB Feature Matching and Homography to warp the auxiliary image to the main perspective.

  - It checks for overlap and re-runs detection on the clean, warped image to replace the broken mask.

5. Parallelization: Uses ThreadPoolExecutor to process all face groups simultaneously.

# Setup

Install Dependencies:
```
pip install -r requirements.txt
```

Note: `opencv-contrib-python` is recommended for full feature access.

Folder Structure:
Create a `data` folder in the project root. Inside, structure it as follows:
```
/data
    /face1
        image1.jpg
        image2.jpg
    /signface1
        closeup_sign.jpg
    /face2
        ...
    ...
```

Run the System:
```
python sign_pipeline.py
```

# Customization

## Integrating Real AI Models

To replace the mock logic with a real model (e.g., YOLOv8-seg or Mask R-CNN):

Open `sign_pipeline.py`.

Locate the `MockSignDetector` class.

Modify `detect_segmentation` to run your model inference and return binary numpy masks.

## Performance Tuning

Threads: Adjust `max_workers` in `PipelineProcessor.run()` based on your CPU cores.

Stitching: If stitching is too slow, resize input images in `load_images_from_folder` before returning them.