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
        Saves the final stitched/selected reference image and the sign segments.
        The segments are saved as FULL-SIZE PNGs with transparent backgrounds.
        The non-sign area is transparent, but the image dimensions remain equal to main_image.
        """
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

        # Save Main Reference Image
        cv2.imwrite(os.path.join(output_folder, "reference_image.jpg"), main_image)

        # Save Segments
        for idx, mask in enumerate(masks):
            # Ensure mask is uint8 binary
            mask_uint8 = mask.astype(np.uint8)
            
            # Skip empty masks
            if cv2.countNonZero(mask_uint8) == 0:
                continue
                
            # Prepare Alpha Channel
            # If mask is 0/1, scale to 0/255. If it's already 0/255, threshold ensures it.
            # 255 (White) = Opaque (Keep Image), 0 (Black) = Transparent
            _, alpha = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY)
            
            # Split BGR channels of the full-size image
            b, g, r = cv2.split(main_image)
            
            # Merge into BGRA (4 channels)
            # This preserves the full HxW dimensions of the original image
            full_size_png = cv2.merge([b, g, r, alpha])
            
            # Save
            filename = os.path.join(output_folder, f"sign_full_{idx}.png")
            cv2.imwrite(filename, full_size_png)