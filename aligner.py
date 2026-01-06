import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ImageAligner:
    """Handles the 'wrapping' (homography/perspective transform) process."""

    def __init__(self):
        # ORB is fast and efficient for backend processing compared to SIFT
        self.orb = cv2.ORB_create(nfeatures=2000)
        # Hamming distance for binary descriptors (ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_image(self, target_img: np.ndarray, source_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Warps source_img (signface) to match the perspective of target_img (face).
        Returns: (warped_image, homography_matrix)
        """
        # Convert to grayscale for feature detection
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        # Detect features
        kp1, des1 = self.orb.detectAndCompute(gray_target, None)
        kp2, des2 = self.orb.detectAndCompute(gray_source, None)

        if des1 is None or des2 is None:
            return None, None

        # Match features
        matches = self.bf.match(des1, des2)
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)

        # Keep top percentage of matches to remove noise
        good_matches = matches[:int(len(matches) * 0.25)]
        
        if len(good_matches) < 4:
            logger.warning("Not enough matches to compute homography.")
            return None, None

        # Extract location of good matches
        pts_target = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography
        h_matrix, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)

        if h_matrix is None:
            return None, None

        # Warp source image to target perspective
        height, width, channels = target_img.shape
        warped_img = cv2.warpPerspective(source_img, h_matrix, (width, height))

        return warped_img, h_matrix