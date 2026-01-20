import cv2
import numpy as np
import torch
import logging
from typing import Tuple, Optional

# Try to import Kornia for Deep Learning matching
try:
    from kornia.feature import LoFTR
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False

logger = logging.getLogger(__name__)

class ImageAligner:
    """
    Aligner supporting:
    - SIFT (Classical, Robust to scale) - Default
    - ORB (Classical, Fast)
    - SURF (Classical, Robust, requires opencv-contrib)
    - LoFTR (Deep Learning, High accuracy, requires GPU)
    """

    def __init__(self, method: str = "sift"):
        self.method = method.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loftr = None
        self.surf = None
        self.orb = None
        self.bf_orb = None
        
        # SIFT Setup (Always available as fallback/base)
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # ORB Setup
        if self.method == "orb":
            logger.info("Initializing ORB model...")
            self.orb = cv2.ORB_create(nfeatures=5000)
            # ORB uses Hamming distance, crossCheck is good for outlier removal
            self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # SURF Setup
        elif self.method == "surf":
            logger.info("Initializing SURF model...")
            try:
                # SURF is in xfeatures2d, might be non-free or missing depending on cv2 version
                self.surf = cv2.xfeatures2d.SURF_create(400)
            except Exception as e:
                logger.warning(f"Could not load SURF (check opencv-contrib-python and non-free license): {e}. Falling back to SIFT.")
                self.method = "sift"

        # LoFTR Setup
        elif self.method == "loftr":
            if HAS_KORNIA and self.device == 'cuda':
                try:
                    logger.info("Initializing LoFTR model...")
                    self.loftr = LoFTR(pretrained="outdoor").to(self.device)
                    logger.info("LoFTR Model Loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load LoFTR: {e}. Falling back to SIFT.")
                    self.method = "sift"
            else:
                logger.warning("LoFTR requested but Kornia not found or no CUDA. Falling back to SIFT.")
                self.method = "sift"

    def align_image(self, target_img: np.ndarray, source_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Warps source_img to match target_img perspective.
        """
        if target_img is None or source_img is None:
            return None, None

        if self.method == "loftr":
            try:
                return self._align_loftr(target_img, source_img)
            except Exception as e:
                logger.error(f"LoFTR alignment failed: {e}. Trying SIFT.")
                return self._align_sift(target_img, source_img)
        
        elif self.method == "orb":
            return self._align_orb(target_img, source_img)
        
        elif self.method == "surf":
            return self._align_surf(target_img, source_img)
        
        else:
            return self._align_sift(target_img, source_img)

    # --- METHODS ---

    def _align_orb(self, target_img, source_img):
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(gray_target, None)
        kp2, des2 = self.orb.detectAndCompute(gray_source, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None, None

        # BFMatcher with Hamming distance
        matches = self.bf_orb.match(des1, des2)
        
        # Sort by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Keep top 15% of matches
        num_good = int(len(matches) * 0.15)
        good = matches[:max(num_good, 10)]

        if len(good) < 4:
            return None, None

        pts_target = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)
        
        if H is None: return None, None

        height, width = target_img.shape[:2]
        warped_img = cv2.warpPerspective(source_img, H, (width, height))
        return warped_img, H

    def _align_surf(self, target_img, source_img):
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.surf.detectAndCompute(gray_target, None)
        kp2, des2 = self.surf.detectAndCompute(gray_source, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return None, None

        # SURF uses float descriptors, so we can use FLANN (Same as SIFT)
        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except Exception:
            return None, None

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                # Lowe's ratio test
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) < 4:
            return None, None

        pts_target = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)
        
        if H is None: return None, None

        height, width = target_img.shape[:2]
        warped_img = cv2.warpPerspective(source_img, H, (width, height))
        return warped_img, H

    def _align_loftr(self, target_img, source_img):
        # Resize logic to fit GPU memory if needed
        inference_dim = 640
        h_t, w_t = target_img.shape[:2]
        h_s, w_s = source_img.shape[:2]
        
        scale_t = inference_dim / max(h_t, w_t)
        scale_s = inference_dim / max(h_s, w_s)

        if scale_t < 1.0:
            t_img_small = cv2.resize(target_img, (0,0), fx=scale_t, fy=scale_t)
        else:
            t_img_small = target_img
            scale_t = 1.0

        if scale_s < 1.0:
            s_img_small = cv2.resize(source_img, (0,0), fx=scale_s, fy=scale_s)
        else:
            s_img_small = source_img
            scale_s = 1.0

        img0 = torch.from_numpy(cv2.cvtColor(t_img_small, cv2.COLOR_BGR2GRAY)).float()[None, None].to(self.device) / 255.0
        img1 = torch.from_numpy(cv2.cvtColor(s_img_small, cv2.COLOR_BGR2GRAY)).float()[None, None].to(self.device) / 255.0

        with torch.no_grad():
            input_dict = {"image0": img0, "image1": img1}
            correspondences = self.loftr(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        if len(mkpts0) < 4: return None, None

        mkpts0 /= scale_t
        mkpts1 /= scale_s

        H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)

        if H is None: return None, None

        height, width = target_img.shape[:2]
        warped_img = cv2.warpPerspective(source_img, H, (width, height))
        
        return warped_img, H

    def _align_sift(self, target_img, source_img):
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        gray_source = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray_target, None)
        kp2, des2 = self.sift.detectAndCompute(gray_source, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2: return None, None

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 4: return None, None

        pts_target = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_source = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_source, pts_target, cv2.RANSAC, 5.0)
        if H is None: return None, None

        height, width = target_img.shape[:2]
        warped_img = cv2.warpPerspective(source_img, H, (width, height))
        return warped_img, H

    def warp_mask(self, mask: np.ndarray, h_matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask is None or h_matrix is None: return None
        try:
            return cv2.warpPerspective(mask, h_matrix, (target_shape[1], target_shape[0]), flags=cv2.INTER_NEAREST)
        except: return None
