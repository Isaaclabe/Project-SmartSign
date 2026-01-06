import os
import cv2
import numpy as np
import glob
import shutil
from typing import List

class ImageUtils:
    """Utilities for image I/O and basic transformations."""
    
    @staticmethod
    def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
        """Loads all images from a folder."""
        images = []
        if not os.path.exists(folder_path):
            return images
            
        # Supported extensions
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                images.append(img)
        return images

    @staticmethod
    def save_results(output_folder: str, main_image: np.ndarray, masks: List[np.ndarray]):
        """Saves the final stitched/selected image and its masks."""
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Save Main Image
        cv2.imwrite(os.path.join(output_folder, "reference_image.jpg"), main_image)

        # Save Masks
        for idx, mask in enumerate(masks):
            # Convert binary mask (0, 1) to visible image (0, 255) if needed
            visible_mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, f"mask_{idx}.png"), visible_mask)