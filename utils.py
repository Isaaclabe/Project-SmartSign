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
        """
        Saves the final stitched/selected reference image and the cropped sign segments.
        The segments are saved as PNGs with transparent backgrounds.
        """
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Save Main Reference Image
        cv2.imwrite(os.path.join(output_folder, "reference_image.jpg"), main_image)

        # Save Cropped Segments
        for idx, mask in enumerate(masks):
            # Ensure mask is uint8 binary
            mask_uint8 = mask.astype(np.uint8)
            
            # Find bounding box of the mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            # Get the bounding box of the largest contour (or union of contours)
            x, y, w, h = cv2.boundingRect(mask_uint8)
            
            if w > 0 and h > 0:
                # 1. Crop the original image (BGR)
                crop_bgr = main_image[y:y+h, x:x+w]
                
                # 2. Crop the mask
                crop_mask = mask_uint8[y:y+h, x:x+w]
                
                # 3. Apply the mask to the image logic
                # We want the background to be transparent.
                # Create Alpha channel: 255 where mask is 1, 0 where mask is 0
                alpha = (crop_mask * 255).astype(np.uint8)
                
                # Split BGR channels
                b, g, r = cv2.split(crop_bgr)
                
                # Merge into BGRA (4 channels)
                crop_png = cv2.merge([b, g, r, alpha])
                
                # Save
                filename = os.path.join(output_folder, f"sign_crop_{idx}.png")
                cv2.imwrite(filename, crop_png)
