import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageStitcher:
    """Handles the stitching of multiple images."""
    
    def __init__(self):
        # cv2.Stitcher_create() is the modern API for OpenCV > 4.0
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS) 

    def stitch(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        if not images:
            return None
        if len(images) == 1:
            return images[0]

        logger.info(f"Attempting to stitch {len(images)} images...")
        status, stitched = self.stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            logger.info("Stitching successful.")
            return stitched
        else:
            logger.warning(f"Stitching failed with error code {status}. Falling back to first image.")
            # Fallback strategy: Return the largest image or just the first one
            return images[0]