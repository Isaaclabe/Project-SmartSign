import os
import cv2
import numpy as np
import logging
import glob
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Import modules
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner
from detector import SAM3SignDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
HF_TOKEN = "your_huggingface_token_here" 
TEXT_PROMPT = "sign"

# The FIRST folder in this list is treated as the REFERENCE.
# Subsequent folders (data2, data3) will be warped to match data1.
DATA_FOLDERS = ["/content/Project-SmartSign/data1", "/content/Project-SmartSign/data2"] 
# ---------------------

class PipelineProcessor:
    def __init__(self, base_dir: str, detector: SAM3SignDetector, reference_base_dir: Optional[str] = None):
        """
        Initializes the processor for a specific directory.
        :param base_dir: Path to the current data folder (e.g., './data2')
        :param detector: Shared SAM3SignDetector instance
        :param reference_base_dir: Path to the reference folder (e.g., './data1'). 
                                   If provided, we try to align images to this reference.
        """
        self.base_dir = base_dir
        self.reference_base_dir = reference_base_dir
        self.detector = detector
        
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        
        # Verify Base Directory
        if not os.path.exists(self.base_dir):
            logger.error(f"The Input Directory '{self.base_dir}' does not exist.")
            self.valid = False
        else:
            self.valid = True

    def process_single_group(self, face_idx: int):
        """
        Main logic flow for a single index.
        """
        face_folder = os.path.join(self.base_dir, f"face{face_idx}")
        signface_folder = os.path.join(self.base_dir, f"signface{face_idx}")
        output_folder = os.path.join(self.base_dir, f"face{face_idx}_signs")

        logger.info(f"[{self.base_dir}] --- Processing Group {face_idx} ---")

        # 1. Load Images
        if not os.path.exists(face_folder):
            return

        face_images = ImageUtils.load_images_from_folder(face_folder)
        if not face_images:
            return

        # 2. Stitching Logic
        main_image = None
        if len(face_images) == 1:
            main_image = face_images[0]
            logger.info(f"[{self.base_dir}] Group {face_idx}: Single image selected.")
        else:
            main_image = self.stitcher.stitch(face_images)
            logger.info(f"[{self.base_dir}] Group {face_idx}: Stitching completed.")

        if main_image is None:
            logger.error(f"[{self.base_dir}] Group {face_idx}: Failed to produce a main image.")
            return

        # --- NEW STEP: ALIGNMENT TO REFERENCE ---
        if self.reference_base_dir:
            # Construct path to the reference image generated previously
            # e.g., ./data1/face1_signs/reference_image.jpg
            ref_img_path = os.path.join(self.reference_base_dir, f"face{face_idx}_signs", "reference_image.jpg")
            
            if os.path.exists(ref_img_path):
                logger.info(f"[{self.base_dir}] Group {face_idx}: Found reference at {ref_img_path}. Attempting alignment...")
                ref_img = cv2.imread(ref_img_path)
                
                if ref_img is not None:
                    # Align current main_image (source) to ref_img (target)
                    warped_img, _ = self.aligner.align_image(target_img=ref_img, source_img=main_image)
                    
                    if warped_img is not None:
                        logger.info(f"[{self.base_dir}] Group {face_idx}: Alignment SUCCESS. Using warped image.")
                        main_image = warped_img
                    else:
                        logger.warning(f"[{self.base_dir}] Group {face_idx}: Alignment FAILED (not enough matches). Using original perspective.")
                else:
                    logger.warning(f"[{self.base_dir}] Group {face_idx}: Could not load reference image.")
            else:
                logger.info(f"[{self.base_dir}] Group {face_idx}: No reference image found (maybe face{face_idx} missing in data1). Skipping alignment.")
        # ----------------------------------------

        # 3. Sign Detection (Segmentation) using SAM3
        logger.info(f"[{self.base_dir}] Group {face_idx}: Running SAM3 inference...")
        detected_masks = self.detector.detect_segmentation(main_image)
        logger.info(f"[{self.base_dir}] Group {face_idx}: Detected {len(detected_masks)} objects.")
        
        final_masks = []
        signface_images = None 

        for i, mask in enumerate(detected_masks):
            # 4. Check Fragmentation (Occlusion)
            if self.detector.is_fragmented(mask):
                logger.info(f"[{self.base_dir}] Group {face_idx}: Mask {i} fragmented. checking signface...")
                
                if signface_images is None:
                    if os.path.exists(signface_folder):
                        signface_images = ImageUtils.load_images_from_folder(signface_folder)
                    else:
                        signface_images = []

                replaced = False
                for s_img in signface_images:
                    # 5. Wrap Image Process for Occlusion
                    warped_s, _ = self.aligner.align_image(target_img=main_image, source_img=s_img)
                    
                    if warped_s is not None:
                        # Simple overlap check
                        warped_gray = cv2.cvtColor(warped_s, cv2.COLOR_BGR2GRAY)
                        _, warped_mask = cv2.threshold(warped_gray, 1, 1, cv2.THRESH_BINARY)
                        
                        overlap = np.logical_and(mask, warped_mask)
                        if np.count_nonzero(mask) > 0 and (np.count_nonzero(overlap) / np.count_nonzero(mask)) > 0.5:
                            logger.info(f"[{self.base_dir}] Group {face_idx}: Occlusion fix found.")
                            
                            # Re-detect on warped patch
                            new_masks = self.detector.detect_segmentation(warped_s)
                            
                            # Find best match
                            best_candidate = None
                            for candidate in new_masks:
                                intersection = np.logical_and(mask, candidate).sum()
                                union = np.logical_or(mask, candidate).sum()
                                iou = intersection / union if union > 0 else 0
                                
                                if not self.detector.is_fragmented(candidate) and iou > 0.1:
                                    best_candidate = candidate
                                    break # take first good one
                            
                            if best_candidate is not None:
                                final_masks.append(best_candidate)
                                replaced = True
                                break 
                
                if not replaced:
                    final_masks.append(mask)
            else:
                final_masks.append(mask)

        # 7. Save Results
        # Note: If we aligned, main_image is now warped, so result is aligned to data1
        ImageUtils.save_results(output_folder, main_image, final_masks)
        logger.info(f"[{self.base_dir}] Group {face_idx}: Results saved.")

    def run(self):
        if not self.valid: return
        indices = [1, 2, 3, 4, 5, 6, 7, 8]
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.process_single_group, indices)

if __name__ == "__main__":
    print("--- Starting Multi-Folder Pipeline ---")
    
    logger.info("Initializing SAM3 Detector (Global)...")
    try:
        global_detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
    except Exception as e:
        logger.error("CRITICAL: Failed to initialize global detector.")
        raise e

    # Track the reference folder (the first one processed)
    primary_ref_dir = None

    for i, folder_path in enumerate(DATA_FOLDERS):
        print(f"\n=================================================")
        print(f" PROCESSING FOLDER: {folder_path}")
        print(f"=================================================")
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder '{folder_path}' not found. Skipping.")
            continue
            
        # If it's the first folder, it sets itself as reference.
        # If it's the second/third, it uses primary_ref_dir.
        if i == 0:
            processor = PipelineProcessor(folder_path, detector=global_detector, reference_base_dir=None)
            primary_ref_dir = folder_path # Set data1 as reference
        else:
            # Pass data1 as the reference base
            processor = PipelineProcessor(folder_path, detector=global_detector, reference_base_dir=primary_ref_dir)
            
        processor.run()
    
    print("\nAll folders processed.")