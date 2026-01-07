import os
import cv2
import numpy as np
import logging
import glob
from concurrent.futures import ThreadPoolExecutor

# Import modules
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner
from detector import SAM3SignDetector

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# Replace with your valid token
HF_TOKEN = "your_huggingface_token_here"
TEXT_PROMPT = "sign"

# List of folders to process
DATA_FOLDERS = ["/content/Project-SmartSign/data1", "/content/Project-SmartSign/data2"]
# ---------------------

class PipelineProcessor:
    def __init__(self, base_dir: str, detector: SAM3SignDetector):
        """
        Initializes the processor for a specific directory.
        :param base_dir: Path to the data folder (e.g., './data1')
        :param detector: An existing, initialized instance of SAM3SignDetector
        """
        self.base_dir = base_dir
        self.detector = detector  # Use the shared detector instance

        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        
        # Verify Base Directory
        if not os.path.exists(self.base_dir):
            logger.warning(
                f"The Input Directory '{self.base_dir}' does not exist. "
                "Skipping this folder."
            )
            self.valid = False
        else:
            self.valid = True

    def process_single_group(self, face_idx: int):
        """
        Main logic flow for a single index (e.g., face1 + signface1).
        """
        face_folder = os.path.join(self.base_dir, f"face{face_idx}")
        signface_folder = os.path.join(self.base_dir, f"signface{face_idx}")
        output_folder = os.path.join(self.base_dir, f"face{face_idx}_signs")

        logger.info(f"[{self.base_dir}] --- Processing Group {face_idx} ---")

        # 1. Load Images
        if not os.path.exists(face_folder):
            logger.warning(f"Folder not found: {face_folder}. Skipping group {face_idx}.")
            return

        face_images = ImageUtils.load_images_from_folder(face_folder)
        if not face_images:
            logger.warning(f"No images found in {face_folder}. Skipping group {face_idx}.")
            return

        # 2. Logic: One image vs Many images (Stitching)
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

        # 3. Sign Detection (Segmentation) using SAM3
        logger.info(f"[{self.base_dir}] Group {face_idx}: Running SAM3 inference...")
        detected_masks = self.detector.detect_segmentation(main_image)
        logger.info(f"[{self.base_dir}] Group {face_idx}: Detected {len(detected_masks)} objects.")
        
        final_masks = []

        # Load signface images only if needed (Lazy loading)
        signface_images = None 

        for i, mask in enumerate(detected_masks):
            # 4. Check Fragmentation (Occlusion)
            if self.detector.is_fragmented(mask):
                logger.info(f"[{self.base_dir}] Group {face_idx}: Mask {i} is fragmented (occluded). Attempting recovery.")
                
                if signface_images is None:
                    if os.path.exists(signface_folder):
                        signface_images = ImageUtils.load_images_from_folder(signface_folder)
                    else:
                        signface_images = []
                        logger.warning(f"[{self.base_dir}] Group {face_idx}: Signface folder not found.")

                replaced = False
                
                # Try to find a better view in signface folder
                for s_img in signface_images:
                    # 5. Wrap Image Process
                    warped_img, h_matrix = self.aligner.align_image(target_img=main_image, source_img=s_img)
                    
                    if warped_img is not None:
                        # 6. Check Overlap
                        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                        _, warped_mask = cv2.threshold(warped_gray, 1, 1, cv2.THRESH_BINARY)
                        
                        # Calculate overlap ratio
                        overlap = np.logical_and(mask, warped_mask)
                        overlap_count = np.count_nonzero(overlap)
                        mask_count = np.count_nonzero(mask)
                        
                        # If the warped image covers at least 50% of the fragmented area, proceed
                        if mask_count > 0 and (overlap_count / mask_count) > 0.5:
                            logger.info(f"[{self.base_dir}] Group {face_idx}: Overlap found. Re-detecting on warped image.")
                            
                            # Run detection on the warped clean image
                            new_masks = self.detector.detect_segmentation(warped_img)
                            
                            best_candidate = None
                            best_iou = 0
                            
                            for candidate in new_masks:
                                intersection = np.logical_and(mask, candidate).sum()
                                union = np.logical_or(mask, candidate).sum()
                                iou = intersection / union if union > 0 else 0
                                
                                # If the new mask is NOT fragmented and overlaps well
                                if not self.detector.is_fragmented(candidate) and iou > 0.1:
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_candidate = candidate
                            
                            if best_candidate is not None:
                                final_masks.append(best_candidate)
                                replaced = True
                                logger.info(f"[{self.base_dir}] Group {face_idx}: Fragmented mask replaced with clean mask.")
                                break 
                
                if not replaced:
                    final_masks.append(mask)
            else:
                final_masks.append(mask)

        # 7. Save Results (Utils will now save cropped PNGs based on previous update)
        ImageUtils.save_results(output_folder, main_image, final_masks)
        logger.info(f"[{self.base_dir}] Group {face_idx}: Processing complete. Results saved to {output_folder}")

    def run(self):
        """
        Runs the pipeline for faces 1 to 8.
        """
        if not self.valid:
            return
        # We have face1 to face8
        indices = [1, 2, 3, 4, 5, 6, 7, 8]
        
        # Parallel Execution for Optimization
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.process_single_group, indices)

if __name__ == "__main__":
    print("--- Starting Multi-Folder Pipeline ---")
    
    # 1. Initialize SAM3 Detector ONCE (Global)
    # This prevents reloading the heavy model for each folder
    logger.info("Initializing SAM3 Detector (Global Shared Instance)...")
    try:
        global_detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
    except Exception as e:
        logger.error("CRITICAL: Failed to initialize global detector.")
        # We exit here because the pipeline cannot run without the model
        raise e

    # 2. Iterate through Data Folders
    for folder_path in DATA_FOLDERS:
        print(f"\n=================================================")
        print(f" PROCESSING FOLDER: {folder_path}")
        print(f"=================================================")
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder '{folder_path}' not found. Skipping.")
            continue
            
        # Create processor for this folder, passing the initialized detector
        processor = PipelineProcessor(folder_path, detector=global_detector)
        processor.run()
    
    print("\nAll folders processed.")
