import os
import cv2
import numpy as np
import logging
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
HF_TOKEN = "your-token"  # Your Token
TEXT_PROMPT = "sign"
WORK_DIR = "./data"
# ---------------------

class PipelineProcessor:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        
        # Initialize SAM3 Detector (This will take time to load weights)
        logger.info("Initializing SAM3 Detector...")
        self.detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)

    def process_single_group(self, face_idx: int):
        """
        Main logic flow for a single index (e.g., face1 + signface1).
        """
        face_folder = os.path.join(self.base_dir, f"face{face_idx}")
        signface_folder = os.path.join(self.base_dir, f"signface{face_idx}")
        output_folder = os.path.join(self.base_dir, f"face{face_idx}_signs")

        logger.info(f"--- Processing Group {face_idx} ---")

        # 1. Load Images
        face_images = ImageUtils.load_images_from_folder(face_folder)
        if not face_images:
            logger.info(f"Skipping face{face_idx} (empty or not found).")
            return

        # 2. Logic: One image vs Many images (Stitching)
        main_image = None
        if len(face_images) == 1:
            main_image = face_images[0]
            logger.info(f"Group {face_idx}: Single image selected.")
        else:
            main_image = self.stitcher.stitch(face_images)
            logger.info(f"Group {face_idx}: Stitching completed.")

        if main_image is None:
            logger.error(f"Group {face_idx}: Failed to produce a main image.")
            return

        # 3. Sign Detection (Segmentation) using SAM3
        logger.info(f"Group {face_idx}: Running SAM3 inference...")
        detected_masks = self.detector.detect_segmentation(main_image)
        logger.info(f"Group {face_idx}: Detected {len(detected_masks)} objects.")
        
        final_masks = []

        # Load signface images only if needed (Lazy loading for optimization)
        signface_images = None 

        for i, mask in enumerate(detected_masks):
            # 4. Check Fragmentation (Occlusion)
            if self.detector.is_fragmented(mask):
                logger.info(f"Group {face_idx}: Mask {i} is fragmented (occluded). Attempting recovery via wrap-process.")
                
                if signface_images is None:
                    signface_images = ImageUtils.load_images_from_folder(signface_folder)

                replaced = False
                
                # Try to find a better view in signface folder
                for s_img in signface_images:
                    # 5. Wrap Image Process
                    warped_img, h_matrix = self.aligner.align_image(target_img=main_image, source_img=s_img)
                    
                    if warped_img is not None:
                        # 6. Check Overlap
                        # Create a bounding box of the fragmented mask in the main image
                        # y_indices, x_indices = np.where(mask > 0) # mask is already numpy uint8
                        
                        # Basic overlap check: Does the warped image cover the fragmented region?
                        # Since warped_img is full size (black where no data), we check non-zero pixels
                        warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                        _, warped_mask = cv2.threshold(warped_gray, 1, 1, cv2.THRESH_BINARY)
                        
                        # Calculate overlap ratio
                        overlap = np.logical_and(mask, warped_mask)
                        overlap_count = np.count_nonzero(overlap)
                        mask_count = np.count_nonzero(mask)
                        
                        # If the warped image covers at least 50% of the fragmented area, proceed
                        if mask_count > 0 and (overlap_count / mask_count) > 0.5:
                            logger.info(f"Group {face_idx}: Overlap found. Re-detecting on warped image.")
                            
                            # Run detection on the warped clean image
                            new_masks = self.detector.detect_segmentation(warped_img)
                            
                            # Find the specific mask in the new results that corresponds to our location
                            # (Heuristic: Find mask with highest IoU with original fragmented mask)
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
                                logger.info(f"Group {face_idx}: Fragmented mask replaced with clean mask.")
                                break # Stop looking through signface images
                
                if not replaced:
                    logger.warning(f"Group {face_idx}: Could not fix fragmentation. Keeping original.")
                    final_masks.append(mask)
            else:
                final_masks.append(mask)

        # 7. Save Results
        ImageUtils.save_results(output_folder, main_image, final_masks)
        logger.info(f"Group {face_idx}: Processing complete. Results saved to {output_folder}")

    def run(self):
        """
        Runs the pipeline for all 4 faces in parallel.
        """
        # We have face1 to face4
        indices = [1, 2, 3, 4] 
        
        # Parallel Execution for Optimization
        # Note: While loading/stitching is parallel, SAM3 inference is locked sequentially
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.process_single_group, indices)

if __name__ == "__main__":
    # Ensure this directory exists or change to your inputs
    
    # Create dummy directories for demonstration if they don't exist
    for i in range(1, 5):
        os.makedirs(os.path.join(WORK_DIR, f"face{i}"), exist_ok=True)
        os.makedirs(os.path.join(WORK_DIR, f"signface{i}"), exist_ok=True)
        
    print(f"Starting pipeline in {WORK_DIR}...")
    processor = PipelineProcessor(WORK_DIR)
    processor.run()
    print("Pipeline finished.")
