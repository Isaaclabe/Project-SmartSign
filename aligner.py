import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class ImageAligner:
    """Handles the 'wrapping' (homography/perspective transform) process."""

    def __init__(self):
        # ORB is fast and efficient for backend processing
        self.orb = cv2.ORB_create(nfeatures=5000) 
        # Hamming distance for binary descriptors (ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def align_image(self, target_img: np.ndarray, source_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Warps source_img to match the perspective of target_img.
        Returns: (warped_image, homography_matrix)
        """
        if target_img is None or source_img is None:
            return None, None

        # Convert to grayscale
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        # Detect features
        kp1, des1 = self.orb.detectAndCompute(gray_target, None)
        kp2, des2 = self.orb.detectAndCompute(gray_source, None)

        if des1 is None or des2 is None:
            # logger.warning("No descriptors found.")
            return None, None

        # Match features
        matches = self.bf.match(des1, des2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Filter top matches (keep top 25% or at least good ones)
        keep_percent = 0.25
        num_keep = int(len(matches) * keep_percent)
        good_matches = matches[:max(num_keep, 10)]
        
        if len(good_matches) < 4:
            # logger.warning(f"Not enough matches ({len(good_matches)}) to compute homography.")
            return None, None

        # Extract location of good matches
        pts_target = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography (Source -> Target)
        h_matrix, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)

        if h_matrix is None:
            return None, None

        # Warp source image
        height, width = target_img.shape[:2]
        warped_img = cv2.warpPerspective(source_img, h_matrix, (width, height))

        return warped_img, h_matrix

    def warp_mask(self, mask: np.ndarray, h_matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Warps a binary mask using the provided homography matrix.
        """
        if mask is None or h_matrix is None:
            return None
            
        warped_mask = cv2.warpPerspective(mask, h_matrix, (target_shape[1], target_shape[0]), flags=cv2.INTER_NEAREST)
        return warped_mask
