import os
import cv2
import numpy as np
import glob
import shutil
from typing import List, Tuple, Optional

class ImageUtils:
    """Utilities for image I/O, directory management, and geometry."""
    
    @staticmethod
    def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
        """Loads all images from a folder."""
        images = []
        if not os.path.exists(folder_path):
            return images
            
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = sorted(glob.glob(os.path.join(folder_path, '*')))
        
        # Filter by extension manually to handle case sensitivity if needed, 
        # or just rely on glob. Using glob loop for safety.
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Sort to ensure consistent order (e.g. image1, image2)
        image_files = sorted(list(set(image_files))) 

        for f in image_files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
        return images

    @staticmethod
    def clear_and_create_dir(path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    @staticmethod
    def ensure_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_crop(image: np.ndarray, mask: np.ndarray, save_path: str):
        """
        Saves a crop of the image defined by the mask.
        The background is made transparent.
        """
        if image is None or mask is None:
            return

        # Ensure mask is binary uint8
        mask = mask.astype(np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
            
        # Get bounding box
        coords = cv2.findNonZero(mask)
        if coords is None:
            return
            
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop
        crop_img = image[y:y+h, x:x+w]
        crop_mask = mask[y:y+h, x:x+w]
        
        # Create BGRA
        b, g, r = cv2.split(crop_img)
        # Alpha is 255 where mask is 1, else 0
        alpha = crop_mask * 255
        
        out = cv2.merge([b, g, r, alpha])
        cv2.imwrite(save_path, out)

    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculates Intersection over Union between two binary masks."""
        if mask1.shape != mask2.shape:
            # Resize mask2 to match mask1 if dimensions differ (robustness)
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
        """Merges a list of masks into one."""
        if not masks:
            return None
        final_mask = np.zeros_like(masks[0])
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        return final_mask.astype(np.uint8)
