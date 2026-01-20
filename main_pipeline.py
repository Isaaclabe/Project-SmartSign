import os
import cv2
import glob
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

# Local imports
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner
from detector import SAM3SignDetector
from VLM_change import VLMProcessor

# --- CONFIGURATION ---
HF_TOKEN = "your_huggingface_token_here"
GOOGLE_API_KEY = "your_google_api_key_here"
ROOT_DIR = "/content/Project-SmartSign/data-image"
TEXT_PROMPT = "sign"
IOU_THRESHOLD_PAIRING = 0.50
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignPipeline:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        
        # Initialize Detectors and VLM
        try:
            self.detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
        except Exception as e:
            logger.error(f"Failed to init SAM3: {e}")
            raise e

        self.vlm = VLMProcessor(api_key=GOOGLE_API_KEY)

    # =========================================================================
    # STEP 1: Extract Signs from Face (Wide View)
    # =========================================================================
    def process_face_extraction(self, store_path: str, data_folder: str, face_idx: str):
        """
        Stitches face images, detects signs, and returns the Main Image + Masks.
        """
        face_path = os.path.join(store_path, data_folder, f"face{face_idx}")
        if not os.path.exists(face_path):
            return None, []

        images = ImageUtils.load_images_from_folder(face_path)
        if not images:
            return None, []

        # Stitch
        if len(images) > 1:
            main_image = self.stitcher.stitch(images)
        else:
            main_image = images[0]

        if main_image is None:
            return None, []

        # Detect
        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    # =========================================================================
    # STEP 2: Extract Signs from Signface (Baseline/Close-up View)
    # =========================================================================
    def process_baseline_extraction(self, store_path: str, data_folder: str, face_idx: str):
        """
        Loads signface image (close-up), detects signs, and returns Image + Masks.
        """
        signface_path = os.path.join(store_path, data_folder, f"signface{face_idx}")
        if not os.path.exists(signface_path):
            return None, []

        images = ImageUtils.load_images_from_folder(signface_path)
        if not images:
            return None, []
        
        # Usually signface is a single close-up, but stitch if needed
        main_image = images[0] 
        
        # Detect
        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    # =========================================================================
    # STEP 3 & 4 & 5: Filter, Pair, Merge
    # =========================================================================
    def run_store_pipeline(self, store_path: str):
        store_name = os.path.basename(store_path)
        logger.info(f"=== Processing Store: {store_name} ===")

        # Identify data folders (data1, data2, ...)
        data_folders = sorted([os.path.basename(p) for p in glob.glob(os.path.join(store_path, "data*"))])
        if "data1" not in data_folders or "data2" not in data_folders:
            logger.warning(f"Store {store_name} missing data1 or data2. Skipping pairing.")
            return

        # Prepare storage for detected data per face
        # Structure: detections[data_folder][face_idx] = {'image': img, 'masks': [m1, m2], 'baseline_masks': [bm1]}
        detections = {d: {} for d in data_folders}

        # --- A. DETECT & FILTER (Steps 1, 2, 3) ---
        for data_name in data_folders:
            # Create Output Directories
            det_sign_dir = os.path.join(store_path, "detected_sign", data_name)
            det_base_dir = os.path.join(store_path, "detected_baseline_sign", data_name)
            ImageUtils.clear_and_create_dir(det_sign_dir)
            ImageUtils.clear_and_create_dir(det_base_dir)

            # Process Faces 1..8
            for i in range(1, 9):
                face_idx = str(i)
                
                # 1. Get Wide Detections (Face)
                face_img, face_masks = self.process_face_extraction(store_path, data_name, face_idx)
                
                # 2. Get Baseline Detections (Signface)
                base_img, base_masks = self.process_baseline_extraction(store_path, data_name, face_idx)

                if face_img is None: continue

                # Save Baseline Crops (Step 2 requirement)
                if base_img is not None and base_masks:
                    for bi, bmask in enumerate(base_masks):
                        save_p = os.path.join(det_base_dir, f"face{face_idx}_b{bi}.png")
                        ImageUtils.save_crop(base_img, bmask, save_p)

                # 3. Filter Face Masks using Baseline (Step 3 requirement)
                valid_face_masks = []
                
                if base_img is not None and base_masks:
                    # Align Baseline Image -> Face Image
                    warped_base, H = self.aligner.align_image(face_img, base_img)
                    
                    if H is not None:
                        # Transform baseline masks to face coordinates
                        warped_base_masks = [self.aligner.warp_mask(m, H, face_img.shape) for m in base_masks]
                        
                        # Check each face mask against any warped baseline mask
                        for fm in face_masks:
                            keep = False
                            for wbm in warped_base_masks:
                                if wbm is None: continue
                                # Check for significant overlap
                                iou = ImageUtils.calculate_iou(fm, wbm)
                                # Overlap check: simply if intersection > 0 or IoU > threshold
                                # Using simple intersection ratio here
                                intersection = np.logical_and(fm, wbm).sum()
                                if intersection > 0: 
                                    keep = True
                                    break
                            if keep:
                                valid_face_masks.append(fm)
                    else:
                        # Alignment failed, keep all or none? 
                        # Conservative: Keep all if we can't filter
                        valid_face_masks = face_masks
                else:
                    # No baseline exists (no signface folder), assume valid or skip?
                    # Assuming we keep them if they were detected
                    valid_face_masks = face_masks

                # Save Valid Face Crops (Step 1 requirement)
                for vi, vmask in enumerate(valid_face_masks):
                    save_p = os.path.join(det_sign_dir, f"face{face_idx}_s{vi}.png")
                    ImageUtils.save_crop(face_img, vmask, save_p)

                # Store in memory for Pairing Step
                detections[data_name][face_idx] = {
                    'image': face_img,
                    'masks': valid_face_masks
                }

        # --- B. PAIRING (Step 4 & 5) ---
        # Pair data1 (Ref) vs data2 (Target)
        pairs_dir_d1 = os.path.join(store_path, "pairs", "data1")
        pairs_dir_d2 = os.path.join(store_path, "pairs", "data2")
        ImageUtils.clear_and_create_dir(pairs_dir_d1)
        ImageUtils.clear_and_create_dir(pairs_dir_d2)

        ref_data = detections['data1']
        tgt_data = detections['data2']

        # Store candidate pairs: list of (iou, ref_crop, tgt_crop, ref_mask_area)
        candidate_pairs = []

        for face_idx in ref_data.keys():
            if face_idx not in tgt_data: continue
            
            ref_entry = ref_data[face_idx]
            tgt_entry = tgt_data[face_idx]
            
            # Align Target Image -> Reference Image
            warped_tgt_img, H = self.aligner.align_image(ref_entry['image'], tgt_entry['image'])
            
            if H is None:
                logger.warning(f"Could not align face{face_idx} between data1 and data2.")
                continue

            # Transform Target Masks -> Reference Coordinates
            warped_tgt_masks = []
            for tm in tgt_entry['masks']:
                wm = self.aligner.warp_mask(tm, H, ref_entry['image'].shape)
                warped_tgt_masks.append(wm)

            # Compare Ref Masks vs Warped Target Masks
            for r_i, r_mask in enumerate(ref_entry['masks']):
                best_iou = 0
                best_t_idx = -1
                
                for t_i, wt_mask in enumerate(warped_tgt_masks):
                    if wt_mask is None: continue
                    iou = ImageUtils.calculate_iou(r_mask, wt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_i
                
                # Step 4: Threshold Check
                if best_iou > IOU_THRESHOLD_PAIRING:
                    # We have a candidate pair
                    candidate_pairs.append({
                        'face': face_idx,
                        'iou': best_iou,
                        'ref_mask': r_mask,
                        'tgt_mask_warped': warped_tgt_masks[best_t_idx],
                        'tgt_mask_orig': tgt_entry['masks'][best_t_idx], # For cropping original
                        'ref_img': ref_entry['image'],
                        'tgt_img': tgt_entry['image'], # Crop from original, not warped
                        'id_ref': r_i,
                        'id_tgt': best_t_idx
                    })

        # --- C. MERGE / NMS (Step 5) ---
        # Logic: If two candidate pairs overlap significantly in the REFERENCE frame, 
        # keep the one with higher pairing IoU (better match) or larger area.
        
        final_pairs = []
        # Sort by IoU descending (prioritize best matches)
        candidate_pairs.sort(key=lambda x: x['iou'], reverse=True)
        
        kept_indices = []
        
        for i, cand in enumerate(candidate_pairs):
            is_duplicate = False
            for k_idx in kept_indices:
                existing = candidate_pairs[k_idx]
                
                # Check overlap between the Reference masks of the two pairs
                overlap = ImageUtils.calculate_iou(cand['ref_mask'], existing['ref_mask'])
                
                # If they overlap significantly (e.g. > 0.3), it's the same sign detected twice (coarse vs fine)
                # Since we sorted by Pairing IoU, the existing one is "better aligned"
                if overlap > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
                final_pairs.append(cand)

        # Save Final Pairs
        vlm_tasks = []

        for idx, pair in enumerate(final_pairs):
            pair_name = f"pair_{idx}_face{pair['face']}"
            
            p1_path = os.path.join(pairs_dir_d1, f"{pair_name}.png")
            p2_path = os.path.join(pairs_dir_d2, f"{pair_name}.png")
            
            # Save Ref Crop
            ImageUtils.save_crop(pair['ref_img'], pair['ref_mask'], p1_path)
            
            # Save Target Crop (from original Target image, using original mask)
            ImageUtils.save_crop(pair['tgt_img'], pair['tgt_mask_orig'], p2_path)
            
            vlm_tasks.append((p1_path, p2_path))

        logger.info(f"Store {store_name}: Saved {len(final_pairs)} aligned pairs.")

        # --- D. VLM ANALYSIS (Step 6) ---
        if vlm_tasks:
            logger.info(f"Starting VLM Analysis for {len(vlm_tasks)} pairs...")
            report_path = os.path.join(store_path, "VLM_Report.txt")
            
            with open(report_path, "w") as f:
                f.write(f"VLM Report for {store_name}\n")
                f.write("="*60 + "\n")
                
                for p1, p2 in vlm_tasks:
                    logger.info(f"Analyzing {os.path.basename(p1)}...")
                    result = self.vlm.analyze_pair(p1, p2)
                    
                    f.write(f"\nPAIR: {os.path.basename(p1)}\n")
                    f.write("-" * 20 + "\n")
                    f.write(result + "\n")
                    f.write("="*60 + "\n")
            
            logger.info(f"VLM Report saved to {report_path}")

    def run(self):
        store_paths = sorted(glob.glob(os.path.join(self.root_dir, "store*")))
        for store in store_paths:
            self.run_store_pipeline(store)

if __name__ == "__main__":
    if not os.path.exists(ROOT_DIR):
        print("Root directory not found.")
        exit(1)
        
    pipeline = SignPipeline(ROOT_DIR)
    pipeline.run()
