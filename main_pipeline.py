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
ALIGNMENT_METHOD = "sift" # Options: "sift", "loftr", "orb", "surf"
KEEP_ONLY_BIGGEST_BASELINE_MASK = True # If True, keeps only the largest sign detected in baseline images
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
    def __init__(self, root_dir: str, debug: bool = False, align_method: str = "sift"):
        self.root_dir = root_dir
        self.debug = debug
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner(method=align_method)
        self.plot_counter = 0 # Unique ID for saved plots
        
        print(f"\n{Colors.HEADER}--- Initializing Models ---{Colors.ENDC}")
        print(f"Alignment Method: {Colors.BOLD}{align_method.upper()}{Colors.ENDC}")
        print(f"Keep Only Biggest Baseline: {Colors.BOLD}{KEEP_ONLY_BIGGEST_BASELINE_MASK}{Colors.ENDC}")
        
        try:
            self.detector = SAM3SignDetector(hf_token=HF_TOKEN, text_prompt=TEXT_PROMPT)
        except Exception as e:
            logger.error(f"Failed to init SAM3: {e}")

        self.vlm = VLMProcessor(api_key=GOOGLE_API_KEY)

    # --- VISUALIZATION HELPERS ---
    def safe_overlay(self, img, mask, color_bgr):
        """Safely overlays a colored mask on an image with a Magenta contour."""
        if img is None: return np.zeros((100,100,3), dtype=np.uint8)
        
        vis = img.copy()
        if mask is None: return vis

        try:
            # 1. Resize mask to match image
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 2. Squeeze extra dimensions (e.g. (H,W,1) -> (H,W))
            if mask.ndim == 3:
                mask = mask.squeeze()
                
            # 3. Boolean indexing for Fill
            mask_bool = mask > 0
            
            if np.any(mask_bool):
                # Fill
                roi = vis[mask_bool]
                color_arr = np.array(color_bgr, dtype=np.float32)
                # Blend: 60% original image + 40% color
                blended = (roi.astype(np.float32) * 0.6 + color_arr * 0.4).astype(np.uint8)
                vis[mask_bool] = blended
                
                # Outline (Contour) in Pure Magenta (BGR: 255, 0, 255)
                # Ensure mask is binary uint8 for findContours
                mask_uint8 = (mask_bool.astype(np.uint8)) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, contours, -1, (255, 0, 255), 2) # Thickness 2

        except Exception as e:
            print(f"{Colors.FAIL}[Plot Error] Overlay failed: {e}{Colors.ENDC}")
            
        return vis

    def debug_plot_detection_warp(self, face_img, face_masks, base_img, base_masks, warped_base, warped_base_masks, title, save_dir):
        """Plots detections and warping logic (Baseline -> Face alignment)."""
        if not self.debug: return
        
        try:
            self.plot_counter += 1
            filename = f"{self.plot_counter:04d}_scan_warp.png"
            save_path = os.path.join(save_dir, filename)

            # Create a 2x2 Grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(title, fontsize=12)
            
            # --- 1. Top-Left: Face Detections (Raw) ---
            vis_face = face_img.copy()
            for m in face_masks:
                vis_face = self.safe_overlay(vis_face, m, (255, 255, 0)) # Cyan
            axes[0, 0].imshow(cv2.cvtColor(vis_face, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("1. Face Detections (Raw)", fontsize=10)
            axes[0, 0].axis('off')

            # --- 2. Top-Right: Baseline Image (Raw) ---
            if base_img is not None:
                vis_base = base_img.copy()
                if base_masks:
                    for m in base_masks:
                        vis_base = self.safe_overlay(vis_base, m, (255, 0, 255)) # Magenta
                axes[0, 1].imshow(cv2.cvtColor(vis_base, cv2.COLOR_BGR2RGB))
                axes[0, 1].set_title("2. Baseline Image (Close-up)", fontsize=10)
            else:
                axes[0, 1].imshow(np.zeros_like(face_img))
                axes[0, 1].set_title("No Baseline Image", fontsize=10)
            axes[0, 1].axis('off')

            # --- 3. Bottom-Left: Warped Baseline Image (Pixels) ---
            if warped_base is not None:
                axes[1, 0].imshow(cv2.cvtColor(warped_base, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title("3. Warped Baseline (Aligned to Face)", fontsize=10)
            else:
                axes[1, 0].imshow(np.zeros_like(face_img))
                axes[1, 0].set_title("Warp Failed / No Base", fontsize=10)
            axes[1, 0].axis('off')

            # --- 4. Bottom-Right: Warp Check (Mask Overlap) ---
            # Show Face image with Warped Baseline Masks (Green) + Face Masks (Red)
            vis_warp = face_img.copy()
            
            if warped_base_masks:
                # Green = "Where the sign SHOULD be" (from close-up)
                for m in warped_base_masks:
                    vis_warp = self.safe_overlay(vis_warp, m, (0, 255, 0))
            
            # Red = "What we detected"
            for m in face_masks:
                 vis_warp = self.safe_overlay(vis_warp, m, (0, 0, 255))

            axes[1, 1].imshow(cv2.cvtColor(vis_warp, cv2.COLOR_BGR2RGB))
            if warped_base_masks:
                axes[1, 1].set_title("4. Filter Check: Green(Base) vs Red(Face)", fontsize=10)
            else:
                axes[1, 1].set_title("4. Filter Check: Alignment Failed", fontsize=10)
            axes[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.show() 
            plt.close(fig)
        except Exception as e:
            print(f"{Colors.FAIL}Could not plot detection/warp: {e}{Colors.ENDC}")

    def debug_plot_pair(self, ref_img, ref_mask, tgt_img, tgt_mask, title, iou, ref_path, tgt_path, save_dir):
        """Plots comparison and saves to disk."""
        if not self.debug: return
        
        try:
            self.plot_counter += 1
            filename = f"{self.plot_counter:04d}_pair_check.png"
            save_path = os.path.join(save_dir, filename)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"{title}\nIoU: {iou:.4f}", fontsize=11)
            
            # Reference
            vis_ref = self.safe_overlay(ref_img, ref_mask, (0, 255, 0)) # Green
            axes[0].imshow(cv2.cvtColor(vis_ref, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"REF: {os.path.basename(ref_path)}")
            axes[0].axis('off')

            # Target
            vis_tgt = self.safe_overlay(tgt_img, tgt_mask, (0, 0, 255)) # Red
            axes[1].imshow(cv2.cvtColor(vis_tgt, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"TGT: {os.path.basename(tgt_path)}")
            axes[1].axis('off')

            plt.savefig(save_path)
            plt.show() 
            plt.close(fig)
        except Exception as e:
            print(f"{Colors.FAIL}Could not plot pair: {e}{Colors.ENDC}")

    def debug_plot_nms(self, ref_img, mask_a, mask_b, title, overlap, name_a, name_b, save_dir):
        """Plots NMS overlap and saves to disk."""
        if not self.debug: return
        
        try:
            self.plot_counter += 1
            filename = f"{self.plot_counter:04d}_nms_check.png"
            save_path = os.path.join(save_dir, filename)
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.set_title(f"{title}\nOverlap: {overlap:.4f} | {name_a} vs {name_b}")
            
            vis = ref_img.copy()
            # Overlay first mask in Green, second in Blue
            vis = self.safe_overlay(vis, mask_a, (0, 255, 0)) 
            vis = self.safe_overlay(vis, mask_b, (255, 0, 0)) 

            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            
            plt.savefig(save_path)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"{Colors.FAIL}Could not plot NMS: {e}{Colors.ENDC}")

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

    def process_baseline_extraction(self, store_path: str, data_folder: str, face_idx: str) -> List[Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Loads ALL signface images (close-up), detects signs, and returns list of (Image, Masks).
        Supports filtering to keep only the largest mask per image.
        """
        signface_path = os.path.join(store_path, data_folder, f"signface{face_idx}")
        if not os.path.exists(signface_path): return []
        images = ImageUtils.load_images_from_folder(signface_path)
        if not images: return []
        
        results = []
        for img in images:
            masks = self.detector.detect_segmentation(img)
            
            if KEEP_ONLY_BIGGEST_BASELINE_MASK and masks:
                # Find largest mask by area (pixel count)
                # np.count_nonzero is safer than sum for binary masks
                largest_mask = max(masks, key=lambda m: np.count_nonzero(m))
                masks = [largest_mask]
            
            results.append((img, masks))
            
        return results

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
                
                # Returns list of tuples: [(img1, masks1), (img2, masks2)...]
                baseline_data = self.process_baseline_extraction(store_path, data_name, face_idx)

                if face_img is None: continue

                # Save Baseline Crops
                baseline_count = 0
                for b_img, b_masks in baseline_data:
                    for bmask in b_masks:
                        save_p = os.path.join(det_base_dir, f"face{face_idx}_b{baseline_count}.png")
                        ImageUtils.save_crop(b_img, bmask, save_p)
                        baseline_count += 1

                # Alignment & Filtering Logic
                valid_face_masks = []
                matched_indices = set()
                any_successful_alignment = False

                if baseline_data:
                    # Check against ALL baseline images
                    for b_idx, (b_img, b_masks) in enumerate(baseline_data):
                        warped_base, H = self.aligner.align_image(face_img, b_img)
                        warped_base_masks = []
                        
                        if H is not None:
                            any_successful_alignment = True
                            warped_base_masks = [self.aligner.warp_mask(m, H, face_img.shape) for m in b_masks]
                            
                            # Check intersections
                            for f_i, fm in enumerate(face_masks):
                                for wbm in warped_base_masks:
                                    if wbm is None: continue
                                    intersection = np.logical_and(fm, wbm).sum()
                                    if intersection > 0: 
                                        matched_indices.add(f_i)
                                        break
                        
                        # Debug plot for this specific baseline image
                        if self.debug:
                            self.debug_plot_detection_warp(
                                face_img, face_masks, 
                                b_img, b_masks,
                                warped_base, warped_base_masks,
                                f"Detection & Warp: {data_name} Face {face_idx} (Base {b_idx})",
                                debug_plot_dir
                            )
                    
                    # Logic: If we successfully aligned at least one baseline image, we only keep matched masks.
                    # If ALL alignments failed (or no baselines), we assume we can't filter, so we keep all (Conservative).
                    if any_successful_alignment:
                        valid_face_masks = [face_masks[i] for i in sorted(list(matched_indices))]
                    else:
                        valid_face_masks = face_masks
                else:
                    # No baseline images exists -> Keep all detected masks
                    valid_face_masks = face_masks

                # Save & Track Valid Signs
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
                            warped_tgt_img, wt_mask, 
                            f"Checking: {ref_fname} vs {tgt_fname}", iou,
                            ref_full_path, tgt_full_path,
                            debug_plot_dir 
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
    pipeline = SignPipeline(ROOT_DIR, debug=DEBUG_MODE, align_method=ALIGNMENT_METHOD)
    pipeline.run()
