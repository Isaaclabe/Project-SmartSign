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
DEBUG_MODE = True  # Set to True to see detailed plots and logs
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI Colors for readable output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m' # Yellow
    FAIL = '\033[91m'    # Red
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SignPipeline:
    def __init__(self, root_dir: str, debug: bool = False):
        self.root_dir = root_dir
        self.debug = debug
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner()
        self.plot_counter = 0 # Unique ID for saved plots
        
        print(f"\n{Colors.HEADER}--- Initializing Models ---{Colors.ENDC}")
        try:
            self.detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
        except Exception as e:
            logger.error(f"Failed to init SAM3: {e}")

        self.vlm = VLMProcessor(api_key=GOOGLE_API_KEY)

    # --- VISUALIZATION HELPERS ---
    def debug_plot_pair(self, ref_img, ref_mask, tgt_img, tgt_mask, title, iou, ref_path, tgt_path, save_dir):
        """Plots comparison and saves to disk."""
        if not self.debug: return
        
        self.plot_counter += 1
        filename = f"{self.plot_counter:04d}_pair_check.png"
        save_path = os.path.join(save_dir, filename)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{title}\nIoU: {iou:.4f}", fontsize=11)
        
        def overlay(img, mask, color_bgr):
            if img is None: return np.zeros((100,100,3), dtype=np.uint8)
            vis = img.copy()
            if mask is not None:
                if mask.shape[:2] != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 0
                if np.any(mask_bool):
                    # Pure numpy blending (Fixes cv2.addWeighted error)
                    roi = vis[mask_bool]
                    color_arr = np.array(color_bgr, dtype=np.float32)
                    # Blend: 60% original image + 40% color
                    blended = (roi.astype(np.float32) * 0.6 + color_arr * 0.4).astype(np.uint8)
                    vis[mask_bool] = blended
            return vis

        # Reference
        vis_ref = overlay(ref_img, ref_mask, (0, 255, 0)) # Green
        axes[0].imshow(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"REF: {os.path.basename(ref_path)}")
        axes[0].axis('off')

        # Target
        vis_tgt = overlay(tgt_img, tgt_mask, (0, 0, 255)) # Red
        axes[1].imshow(cv2.cvtColor(vis_tgt, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"TGT: {os.path.basename(tgt_path)}")
        axes[1].axis('off')

        # Save and Show
        plt.savefig(save_path)
        plt.show() 
        plt.close(fig)

    def debug_plot_nms(self, ref_img, mask_a, mask_b, title, overlap, name_a, name_b, save_dir):
        """Plots NMS overlap and saves to disk."""
        if not self.debug: return
        
        self.plot_counter += 1
        filename = f"{self.plot_counter:04d}_nms_check.png"
        save_path = os.path.join(save_dir, filename)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_title(f"{title}\nOverlap: {overlap:.4f} | {name_a} vs {name_b}")
        
        vis = ref_img.copy()
        
        # Mask A = Green
        if mask_a is not None:
             mask_a_bool = mask_a > 0
             if np.any(mask_a_bool):
                roi = vis[mask_a_bool]
                color = np.array([0, 255, 0], dtype=np.float32)
                vis[mask_a_bool] = (roi.astype(np.float32) * 0.5 + color * 0.5).astype(np.uint8)
             
        # Mask B = Blue
        if mask_b is not None:
             mask_b_bool = mask_b > 0
             if np.any(mask_b_bool):
                roi = vis[mask_b_bool]
                color = np.array([255, 0, 0], dtype=np.float32)
                vis[mask_b_bool] = (roi.astype(np.float32) * 0.5 + color * 0.5).astype(np.uint8)

        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        
        plt.savefig(save_path)
        plt.show()
        plt.close(fig)

    # --- PIPELINE STEPS ---
    def process_face_extraction(self, store_path: str, data_folder: str, face_idx: str):
        face_path = os.path.join(store_path, data_folder, f"face{face_idx}")
        if not os.path.exists(face_path): return None, []
        images = ImageUtils.load_images_from_folder(face_path)
        if not images: return None, []
        if len(images) > 1:
            main_image = self.stitcher.stitch(images)
        else:
            main_image = images[0]
        if main_image is None: return None, []
        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    def process_baseline_extraction(self, store_path: str, data_folder: str, face_idx: str):
        signface_path = os.path.join(store_path, data_folder, f"signface{face_idx}")
        if not os.path.exists(signface_path): return None, []
        images = ImageUtils.load_images_from_folder(signface_path)
        if not images: return None, []
        main_image = images[0] 
        masks = self.detector.detect_segmentation(main_image)
        return main_image, masks

    def plot_results(self, pair_list: List[Dict], results: List[str], save_dir: str):
        if not pair_list: return
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.HEADER}VISUALIZATION: Found {len(pair_list)} Final Pairs{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        for i, (pair, text_report) in enumerate(zip(pair_list, results)):
            path_ref = pair['path_ref']
            path_tgt = pair['path_tgt']
            img1 = cv2.imread(path_ref)
            img2 = cv2.imread(path_tgt)
            if img1 is None or img2 is None: continue
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.suptitle(f"Final Pair {i+1} (Face {pair['face']})", fontsize=14)
            axes[0].imshow(img1)
            axes[0].set_title(f"Ref: {os.path.basename(path_ref)}")
            axes[0].axis('off')
            axes[1].imshow(img2)
            axes[1].set_title(f"Tgt: {os.path.basename(path_tgt)}")
            axes[1].axis('off')
            plt.tight_layout()
            
            # Save final result plot
            save_path = os.path.join(save_dir, f"Final_Result_Pair_{i}.png")
            plt.savefig(save_path)
            plt.show()
            plt.close(fig)
            
            print(f"\n{Colors.CYAN}--- VLM Report Pair {i+1} ---{Colors.ENDC}")
            print(text_report.strip())
            print(f"{Colors.CYAN}{'-' * 50}{Colors.ENDC}\n")

    def run_store_pipeline(self, store_path: str):
        store_name = os.path.basename(store_path)
        print(f"\n{Colors.HEADER}#################################################{Colors.ENDC}")
        print(f"{Colors.HEADER} PROCESSING STORE: {store_name}{Colors.ENDC}")
        print(f"{Colors.HEADER}#################################################{Colors.ENDC}")

        data_folders = sorted([os.path.basename(p) for p in glob.glob(os.path.join(store_path, "data*"))])
        if "data1" not in data_folders or "data2" not in data_folders:
            print(f"{Colors.WARNING}Skipping {store_name}: Missing data1 or data2.{Colors.ENDC}")
            return

        # Setup Debug Plot Directory
        debug_plot_dir = os.path.join(store_path, "debug_plots")
        ImageUtils.clear_and_create_dir(debug_plot_dir)
        print(f"  > {Colors.CYAN}Debug plots will be saved to: {debug_plot_dir}{Colors.ENDC}")

        detections = {d: {} for d in data_folders}

        # --- STEP 1, 2, 3: Detection & Filtering ---
        for data_name in data_folders:
            print(f"  > {Colors.CYAN}Scanning {data_name}...{Colors.ENDC}")
            det_sign_dir = os.path.join(store_path, "detected_sign", data_name)
            det_base_dir = os.path.join(store_path, "detected_baseline_sign", data_name)
            ImageUtils.clear_and_create_dir(det_sign_dir)
            ImageUtils.clear_and_create_dir(det_base_dir)

            for i in range(1, 9):
                face_idx = str(i)
                face_img, face_masks = self.process_face_extraction(store_path, data_name, face_idx)
                base_img, base_masks = self.process_baseline_extraction(store_path, data_name, face_idx)

                if face_img is None: continue

                # Baseline logic
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

                # Save & Track
                crop_filenames = []
                for vi, vmask in enumerate(valid_face_masks):
                    fname = f"face{face_idx}_s{vi}.png"
                    save_p = os.path.join(det_sign_dir, fname)
                    ImageUtils.save_crop(face_img, vmask, save_p)
                    crop_filenames.append(fname)

                # STORE FULL DIR PATH FOR DEBUGGING
                detections[data_name][face_idx] = {
                    'image': face_img, 
                    'masks': valid_face_masks,
                    'filenames': crop_filenames,
                    'dir_path': det_sign_dir
                }

        # --- STEP 4 & 5: Pairing & Merging ---
        print(f"  > {Colors.BLUE}Pairing Data1 vs Data2...{Colors.ENDC}")
        pairs_dir_d1 = os.path.join(store_path, "pairs", "data1")
        pairs_dir_d2 = os.path.join(store_path, "pairs", "data2")
        ImageUtils.clear_and_create_dir(pairs_dir_d1)
        ImageUtils.clear_and_create_dir(pairs_dir_d2)

        ref_data = detections.get('data1', {})
        tgt_data = detections.get('data2', {})
        candidate_pairs = []

        for face_idx in ref_data.keys():
            if face_idx not in tgt_data: continue
            
            ref_entry = ref_data[face_idx]
            tgt_entry = tgt_data[face_idx]
            
            # --- ALIGNMENT ---
            warped_tgt_img, H = self.aligner.align_image(ref_entry['image'], tgt_entry['image'])
            
            if H is None:
                if self.debug: print(f"    [{Colors.FAIL}DEBUG{Colors.ENDC}] Face {face_idx}: Alignment FAILED.")
                continue

            warped_tgt_masks = []
            for tm in tgt_entry['masks']:
                wm = self.aligner.warp_mask(tm, H, ref_entry['image'].shape)
                warped_tgt_masks.append(wm)

            # --- MATCHING MASKS ---
            for r_i, r_mask in enumerate(ref_entry['masks']):
                ref_fname = ref_entry['filenames'][r_i]
                ref_full_path = os.path.join(ref_entry['dir_path'], ref_fname)
                
                best_iou = 0
                best_t_idx = -1
                best_t_path = "None"
                best_wt_mask = None
                
                for t_i, wt_mask in enumerate(warped_tgt_masks):
                    if wt_mask is None: continue
                    tgt_fname = tgt_entry['filenames'][t_i]
                    tgt_full_path = os.path.join(tgt_entry['dir_path'], tgt_fname)
                    
                    iou = ImageUtils.calculate_iou(r_mask, wt_mask)
                    
                    if self.debug:
                        # PLOT EVERY COMPARISON
                        self.debug_plot_pair(
                            ref_entry['image'], r_mask, 
                            warped_tgt_img, wt_mask, # Show warped target for visual alignment check
                            f"Checking: {ref_fname} vs {tgt_fname}", iou,
                            ref_full_path, tgt_full_path,
                            debug_plot_dir # Pass save directory
                        )

                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_i
                        best_t_path = tgt_full_path
                        best_wt_mask = wt_mask
                
                if best_iou > IOU_THRESHOLD_PAIRING:
                    if self.debug:
                        print(f"    [{Colors.GREEN}MATCH{Colors.ENDC}] {Colors.BOLD}{ref_full_path}{Colors.ENDC} <--> {best_t_path} (IoU: {Colors.GREEN}{best_iou:.4f}{Colors.ENDC})")
                        
                    candidate_pairs.append({
                        'face': face_idx,
                        'iou': best_iou,
                        'ref_mask': r_mask,
                        'tgt_mask_orig': tgt_entry['masks'][best_t_idx],
                        'ref_img': ref_entry['image'],
                        'tgt_img': tgt_entry['image'],
                        'ref_fname': ref_fname,
                        'ref_full_path': ref_full_path
                    })
                else:
                    if self.debug:
                        print(f"    [{Colors.WARNING}DISCARD{Colors.ENDC}] {ref_full_path} (Max IoU {best_iou:.4f} too low)")

        # --- STEP 5: NMS / Merge Duplicates ---
        if self.debug:
            print(f"\n    [{Colors.BLUE}DEBUG{Colors.ENDC}] Starting NMS on {len(candidate_pairs)} candidates...")
            
        final_pairs_meta = []
        candidate_pairs.sort(key=lambda x: x['iou'], reverse=True)
        kept_indices = []
        
        for i, cand in enumerate(candidate_pairs):
            is_duplicate = False
            for k_idx in kept_indices:
                existing = candidate_pairs[k_idx]
                overlap = ImageUtils.calculate_iou(cand['ref_mask'], existing['ref_mask'])
                
                if self.debug:
                    self.debug_plot_nms(
                        cand['ref_img'], cand['ref_mask'], existing['ref_mask'],
                        "Duplicate Check (NMS)", overlap,
                        cand['ref_fname'], existing['ref_fname'],
                        debug_plot_dir
                    )
                
                if overlap > 0.3:
                    is_duplicate = True
                    if self.debug:
                        print(f"    [{Colors.FAIL}DUPLICATE{Colors.ENDC}] Dropping {cand['ref_full_path']} (Overlap {overlap:.2f} with {existing['ref_full_path']})")
                    break
            
            if not is_duplicate:
                kept_indices.append(i)
                final_pairs_meta.append(cand)
                if self.debug:
                    print(f"    [{Colors.GREEN}KEEP{Colors.ENDC}] {cand['ref_full_path']}")

        # --- SAVE FINAL PAIRS ---
        vlm_tasks = []
        for idx, pair in enumerate(final_pairs_meta):
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

        print(f"  > {Colors.GREEN}Saved {len(vlm_tasks)} unique pairs.{Colors.ENDC}")

        # --- STEP 6: VLM ---
        if vlm_tasks:
            print(f"  > {Colors.CYAN}Starting VLM Analysis...{Colors.ENDC}")
            report_path = os.path.join(store_path, "VLM_Report.txt")
            results_text = []

            with open(report_path, "w") as f:
                f.write(f"VLM Report for {store_name}\n")
                f.write("="*60 + "\n")
                for task in vlm_tasks:
                    print(f"    - Analyzing {os.path.basename(task['path_ref'])}...")
                    res = self.vlm.analyze_pair(task['path_ref'], task['path_tgt'])
                    results_text.append(res)
                    f.write(f"\nPAIR: {os.path.basename(task['path_ref'])}\n")
                    f.write("-" * 20 + "\n")
                    f.write(res + "\n")
                    f.write("="*60 + "\n")

            self.plot_results(vlm_tasks, results_text, debug_plot_dir)
            print(f"  > {Colors.GREEN}Full report saved to {report_path}{Colors.ENDC}")
        else:
            print(f"  > {Colors.WARNING}No pairs found for VLM.{Colors.ENDC}")

    def run(self):
        store_paths = sorted(glob.glob(os.path.join(self.root_dir, "store*")))
        if not store_paths:
            print(f"{Colors.FAIL}No 'store' directories found in {self.root_dir}{Colors.ENDC}")
            return
        for store in store_paths:
            self.run_store_pipeline(store)

if __name__ == "__main__":
    if not os.path.exists(ROOT_DIR):
        print(f"{Colors.FAIL}Root directory {ROOT_DIR} not found.{Colors.ENDC}")
        exit(1)
    pipeline = SignPipeline(ROOT_DIR, debug=DEBUG_MODE)
    pipeline.run()
