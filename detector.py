import os
import cv2
import torch
import numpy as np
import threading
import requests
import shutil
from PIL import Image
from typing import List, Optional
from huggingface_hub import login

# SAM3 Imports
try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("CRITICAL ERROR: SAM3 not found. Please install via: pip install 'git+https://github.com/facebookresearch/sam3.git'")
    raise

class SAM3SignDetector:
    """
    Real AI Model using Facebook's SAM3 for sign detection.
    """
    _instance = None
    _lock = threading.Lock() # Global lock for GPU access

    def __init__(self, hf_token: str, text_prompt: str = "sign"):
        self.hf_token = hf_token
        self.text_prompt = text_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        self._initialize_model()

    def _download_bpe_file(self, dest_path: str):
        """Downloads the missing BPE file from the SAM3 repository."""
        url = "https://github.com/facebookresearch/sam3/raw/main/assets/bpe_simple_vocab_16e6.txt.gz"
        print(f"Downloading BPE file from {url} to {dest_path}...")
        
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download BPE file: {e}")
            raise RuntimeError("Could not acquire BPE file needed for SAM3 text prompts.")

    def _initialize_model(self):
        """Initializes the SAM3 model. Protected by lock during inference, but init happens once."""
        print(f"Initializing SAM3 on {self.device}...")
        
        # 1. AUTHENTICATION
        try:
            print(f"Attempting HuggingFace Login...")
            login(token=self.hf_token)
        except Exception as e:
            print("----------------------------------------------------------------")
            print("CRITICAL AUTH ERROR: HuggingFace Login failed.")
            print(f"Error details: {e}")
            print("Please ensure your HF_TOKEN is valid and you have accepted the license")
            print("for 'facebook/sam3' at https://huggingface.co/facebook/sam3")
            print("----------------------------------------------------------------")
            raise

        # 2. LOCATE/DOWNLOAD BPE ASSETS
        sam3_root = os.path.dirname(sam3.__file__)
        possible_paths = [
            os.path.join(sam3_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"), # Git clone style
            os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),       # Pkg style
            os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")      # Local fallback
        ]
        
        bpe_path = None
        for p in possible_paths:
            if os.path.exists(p):
                bpe_path = p
                break
        
        if bpe_path is None:
            print("BPE asset not found in package. Downloading to local ./assets folder...")
            local_asset_path = os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")
            self._download_bpe_file(local_asset_path)
            bpe_path = local_asset_path

        print(f"Using BPE path: {bpe_path}")

        # 3. BUILD MODEL
        # Optimization settings
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        try:
            self.model = build_sam3_image_model(bpe_path=bpe_path)
        except Exception as e:
            print(f"Error building SAM3 model: {e}")
            raise

        self.model.to(self.device)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        print("SAM3 Initialization Complete.")

    def detect_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Runs SAM3 inference on the image.
        Returns a list of binary masks (numpy arrays of shape (H, W)).
        """
        if self.model is None:
            return []

        # Convert OpenCV BGR to PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Determine dtype
        dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

        # CRITICAL: Use a lock to prevent multiple threads from accessing the GPU model simultaneously
        with SAM3SignDetector._lock:
            with torch.autocast(self.device, dtype=dtype):
                try:
                    # 1. Initialize Inference State
                    inference_state = self.processor.set_image(pil_image)
                    
                    # 2. Reset Prompts
                    self.processor.reset_all_prompts(inference_state)
                    
                    # 3. Apply Text Prompt
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state, 
                        prompt=self.text_prompt
                    )
                    
                    # 4. Extract Masks
                    masks_tensor = None
                    if hasattr(inference_state, 'pred_masks'):
                         masks_tensor = inference_state.pred_masks
                    elif isinstance(inference_state, dict) and 'pred_masks' in inference_state:
                         masks_tensor = inference_state['pred_masks']
                    else:
                        if 'masks' in inference_state:
                             masks_tensor = inference_state['masks']

                    if masks_tensor is None:
                        return []

                    # Process Masks: Tensor -> List of Numpy Arrays
                    output_masks = []
                    
                    # Ensure we are on CPU and numpy
                    if isinstance(masks_tensor, torch.Tensor):
                        masks_np = masks_tensor.detach().cpu().numpy()
                    else:
                        masks_np = masks_tensor

                    # Iterate over detected objects
                    for m in masks_np:
                        # Remove channel dim if present (1, H, W) -> (H, W)
                        if m.ndim == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        
                        if m.dtype != bool and m.max() > 1.0: 
                            binary_mask = (m > 0).astype(np.uint8) # Logits
                        else:
                            binary_mask = m.astype(np.uint8)

                        # Resize if mask resolution differs
                        h_orig, w_orig = image.shape[:2]
                        if binary_mask.shape != (h_orig, w_orig):
                            binary_mask = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                        output_masks.append(binary_mask)
                        
                    return output_masks

                except Exception as e:
                    print(f"Error during SAM3 inference: {e}")
                    import traceback
                    traceback.print_exc()
                    return []

    def is_fragmented(self, mask: np.ndarray) -> bool:
        """
        Checks if a single segmentation mask is fragmented (more than 1 connected component).
        """
        # Find contours or connected components
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        # num_labels includes background (0). So if num_labels > 2, we have >1 foreground object.
        return num_labels > 2
