import os
import cv2
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

# Local imports
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner
from detector import SAM3SignDetector
from vlm_processor import VLMProcessor

# --- CONFIGURATION ---
HF_TOKEN = "your_huggingface_token_here"
GOOGLE_API_KEY = "your_google_api_key_here"
ROOT_DIR = "/content/Project-SmartSign/data-image"
TEXT_PROMPT = "sign"
IOU_THRESHOLD_PAIRING = 0.50
DEBUG_MODE = True  # Set to True to see detailed pairing logs
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignPipeline:
    def __init__(self, root_dir: str, debug: bool = False):
        self.root_dir = root_dir
        self.debug = debug
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        
        print("\n--- Initializing Models ---")
        try:
            self.detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
        except Exception as e:
            logger.error(f"Failed to init SAM3: {e}")
            # raise e # Commented out to allow debugging other parts if needed

        self.vlm = VLMProcessor(api_key=GOOGLE_API_KEY)

    def process_face_extraction(self, store_path: str, data_folder: str, face_idx: str):
        """Stitches face images, detects signs, and returns the Main Image + Masks."""
        face_path = os.path.join(store_path, data_folder, f"face{face_idx}")
        if not os.path.exists(face_path):
            return None, []

        images = ImageUtils.load_images_from_folder(face_path)
        if not images:
            return None, []

        if len(images) > 1:
            main_image = self.stitcher.stitch(images)
        else:
            main_image = images[0]

        if main_image is None:
            return None, []

        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    def process_baseline_extraction(self, store_path: str, data_folder: str, face_idx: str):
        """Loads signface image, detects signs, and returns Image + Masks."""
        signface_path = os.path.join(store_path, data_folder, f"signface{face_idx}")
        if not os.path.exists(signface_path):
            return None, []

        images = ImageUtils.load_images_from_folder(signface_path)
        if not images:
            return None, []
        
        main_image = images[0] 
        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    def plot_results(self, pair_list: List[Dict], results: List[str]):
        """Plots reference, target, and the VLM report inline."""
        if not pair_list:
            print("No pairs to plot.")
            return

        print(f"\n{'='*80}")
        print(f"VISUALIZATION: Found {len(pair_list)} Pairs")
        print(f"{'='*80}")

        for i, (pair, text_report) in enumerate(zip(pair_list, results)):
            path_ref = pair['path_ref']
            path_tgt = pair['path_tgt']
            
            # Load images for matplotlib
            img1 = cv2.imread(path_ref)
            img2 = cv2.imread(path_tgt)
            
            if img1 is None or img2 is None:
                continue

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # Create Plot
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Pair {i+1} (Face {pair['face']})", fontsize=14)
            
            axes[0].imshow(img1)
            axes[0].set_title(f"Ref: {os.path.basename(path_ref)}")
            axes[0].axis('off')
            
            axes[1].imshow(img2)
            axes[1].set_title(f"Tgt: {os.path.basename(path_tgt)}")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Print Report nicely
            print(f"\n--- VLM Report Pair {i+1} ---")
            print(text_report.strip())
            print("-" * 50 + "\n")

    def run_store_pipeline(self, store_path: str):
        store_name = os.path.basename(store_path)
        print(f"\n#################################################")
        print(f" PROCESSING STORE: {store_name}")
        print(f"#################################################")

        data_folders = sorted([os.path.basename(p) for p in glob.glob(os.path.join(store_path, "data*"))])
        if "data1" not in data_folders or "data2" not in data_folders:
            print(f"Skipping {store_name}: Missing data1 or data2.")
            return

        detections = {d: {} for d in data_folders}

        # --- STEP 1, 2, 3: Detection & Filtering ---
        for data_name in data_folders:
            print(f"  > Scanning {data_name}...")
            det_sign_dir = os.path.join(store_path, "detected_sign", data_name)
            det_base_dir = os.path.join(store_path, "detected_baseline_sign", data_name)
            ImageUtils.clear_and_create_dir(det_sign_dir)
            ImageUtils.clear_and_create_dir(det_base_dir)

            for i in range(1, 9):
                face_idx = str(i)
                
                # Extraction
                face_img, face_masks = self.process_face_extraction(store_path, data_name, face_idx)
                base_img, base_masks = self.process_baseline_extraction(store_path, data_name, face_idx)

                if face_img is None: continue

                # Save Baseline
                if base_img is not None and base_masks:
                    for bi, bmask in enumerate(base_masks):
                        save_p = os.path.join(det_base_dir, f"face{face_idx}_b{bi}.png")
                        ImageUtils.save_crop(base_img, bmask, save_p)

                # Filter Face Masks
                valid_face_masks = []
                if base_img is not None and base_masks:
                    warped_base, H = self.aligner.align_image(face_img, base_img)
                    if H is not None:
                        warped_base_masks = [self.aligner.warp_mask(m, H, face_img.shape) for m in base_masks]
                        for fm in face_masks:
                            keep = False
                            for wbm in warped_base_masks:
                                if wbm is None: continue
                                intersection = np.logical_and(fm, wbm).sum()
                                if intersection > 0: 
                                    keep = True
                                    break
                            if keep: valid_face_masks.append(fm)
                    else:
                        valid_face_masks = face_masks
                else:
                    valid_face_masks = face_masks

                # Save Valid Signs and track filenames
                crop_filenames = []
                for vi, vmask in enumerate(valid_face_masks):
                    fname = f"face{face_idx}_s{vi}.png"
                    save_p = os.path.join(det_sign_dir, fname)
                    ImageUtils.save_crop(face_img, vmask, save_p)
                    crop_filenames.append(fname)

                detections[data_name][face_idx] = {
                    'image': face_img, 
                    'masks': valid_face_masks,
                    'filenames': crop_filenames
                }

        # --- STEP 4 & 5: Pairing & Merging ---
        print("  > Pairing Data1 vs Data2...")
        pairs_dir_d1 = os.path.join(store_path, "pairs", "data1")
        pairs_dir_d2 = os.path.join(store_path, "pairs", "data2")
        ImageUtils.clear_and_create_dir(pairs_dir_d1)
        ImageUtils.clear_and_create_dir(pairs_dir_d2)

        ref_data = detections.get('data1', {})
        tgt_data = detections.get('data2', {})
        candidate_pairs = []

        for face_idx in ref_data.keys():
            if face_idx not in tgt_data:
                if self.debug:
                    print(f"    [DEBUG] Face {face_idx}: Present in Data1 but missing in Data2. Skipping.")
                continue
            
            ref_entry = ref_data[face_idx]
            tgt_entry = tgt_data[face_idx]
            
            if self.debug:
                print(f"    [DEBUG] Face {face_idx}: Attempting alignment...")

            warped_tgt_img, H = self.aligner.align_image(ref_entry['image'], tgt_entry['image'])
            
            if H is None:
                if self.debug:
                    print(f"    [DEBUG] Face {face_idx}: Alignment FAILED (Not enough matches).")
                continue

            warped_tgt_masks = []
            for tm in tgt_entry['masks']:
                wm = self.aligner.warp_mask(tm, H, ref_entry['image'].shape)
                warped_tgt_masks.append(wm)

            # Compare Masks
            for r_i, r_mask in enumerate(ref_entry['masks']):
                ref_name = ref_entry['filenames'][r_i]
                best_iou = 0
                best_t_idx = -1
                best_t_name = "None"
                
                for t_i, wt_mask in enumerate(warped_tgt_masks):
                    if wt_mask is None: continue
                    tgt_name = tgt_entry['filenames'][t_i]
                    
                    iou = ImageUtils.calculate_iou(r_mask, wt_mask)
                    
                    if self.debug and iou > 0.1:
                        print(f"    [DEBUG] Face {face_idx}: {ref_name} vs {tgt_name} -> IoU: {iou:.4f}")
                        
                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_i
                        best_t_name = tgt_name
                
                if best_iou > IOU_THRESHOLD_PAIRING:
                    if self.debug:
                        print(f"    [DEBUG] Face {face_idx}: MATCH FOUND! {ref_name} paired with {best_t_name} (IoU: {best_iou:.4f})")
                        
                    candidate_pairs.append({
                        'face': face_idx,
                        'iou': best_iou,
                        'ref_mask': r_mask,
                        'tgt_mask_orig': tgt_entry['masks'][best_t_idx],
                        'ref_img': ref_entry['image'],
                        'tgt_img': tgt_entry['image'],
                        'ref_fname': ref_name,
                        'tgt_fname': best_t_name
                    })
                else:
                    if self.debug:
                        print(f"    [DEBUG] Face {face_idx}: {ref_name} discarded. Best IoU {best_iou:.4f} < Threshold {IOU_THRESHOLD_PAIRING}")

        # NMS / Merge Duplicates
        if self.debug:
            print(f"    [DEBUG] Starting Merge/NMS on {len(candidate_pairs)} candidates...")
            
        final_pairs_meta = []
        candidate_pairs.sort(key=lambda x: x['iou'], reverse=True)
        kept_indices = []
        
        for i, cand in enumerate(candidate_pairs):
            is_duplicate = False
            for k_idx in kept_indices:
                existing = candidate_pairs[k_idx]
                overlap = ImageUtils.calculate_iou(cand['ref_mask'], existing['ref_mask'])
                
                if overlap > 0.3:
                    is_duplicate = True
                    if self.debug:
                        print(f"    [DEBUG] Dropping Duplicate: {cand['ref_fname']} (IoU {cand['iou']:.2f}) overlaps {overlap:.2f} with kept pair {existing['ref_fname']}.")
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
                final_pairs_meta.append(cand)
                if self.debug:
                    print(f"    [DEBUG] Keeping: {cand['ref_fname']} (IoU {cand['iou']:.4f})")

        # Save Final Pairs
        vlm_tasks = []
        for idx, pair in enumerate(final_pairs_meta):
            # Construct meaningful pair name
            # pair_X_faceY_refZ.png
            pair_name = f"pair_{idx}_{pair['ref_fname'].replace('.png','')}"
            
            p1_path = os.path.join(pairs_dir_d1, f"{pair_name}.png")
            p2_path = os.path.join(pairs_dir_d2, f"{pair_name}.png")
            
            ImageUtils.save_crop(pair['ref_img'], pair['ref_mask'], p1_path)
            ImageUtils.save_crop(pair['tgt_img'], pair['tgt_mask_orig'], p2_path)
            
            vlm_tasks.append({
                'path_ref': p1_path,
                'path_tgt': p2_path,
                'face': pair['face']
            })

        print(f"  > Saved {len(vlm_tasks)} unique pairs.")

        # --- STEP 6: VLM Analysis & Visualization ---
        if vlm_tasks:
            print(f"  > Starting VLM Analysis for {len(vlm_tasks)} pairs...")
            report_path = os.path.join(store_path, "VLM_Report.txt")
            results_text = []

            with open(report_path, "w") as f:
                f.write(f"VLM Report for {store_name}\n")
                f.write("="*60 + "\n")
                
                for task in vlm_tasks:
                    print(f"    - Analyzing Pair Face {task['face']} ({os.path.basename(task['path_ref'])})...")
                    res = self.vlm.analyze_pair(task['path_ref'], task['path_tgt'])
                    results_text.append(res)
                    
                    f.write(f"\nPAIR: {os.path.basename(task['path_ref'])}\n")
                    f.write("-" * 20 + "\n")
                    f.write(res + "\n")
                    f.write("="*60 + "\n")

            # PLOT THE RESULTS
            self.plot_results(vlm_tasks, results_text)
            print(f"  > Full report saved to {report_path}")
        else:
            print("  > No pairs found for VLM.")

    def run(self):
        store_paths = sorted(glob.glob(os.path.join(self.root_dir, "store*")))
        if not store_paths:
            print(f"No 'store' directories found in {self.root_dir}")
            return
            
        for store in store_paths:
            self.run_store_pipeline(store)

if __name__ == "__main__":
    if not os.path.exists(ROOT_DIR):
        print(f"Root directory {ROOT_DIR} not found.")
        exit(1)
        
    pipeline = SignPipeline(ROOT_DIR, debug=DEBUG_MODE)
    pipeline.run()
