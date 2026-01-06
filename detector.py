import os
import cv2
import torch
import numpy as np
import threading
from PIL import Image
from typing import List
from huggingface_hub import login

# SAM3 Imports
# Note: Ensure 'sam3' is installed via the git command in requirements
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

    def _initialize_model(self):
        """Initializes the SAM3 model. Protected by lock during inference, but init happens once."""
        print(f"Initializing SAM3 on {self.device}...")
        
        try:
            login(token=self.hf_token)
        except Exception as e:
            print(f"Warning: HuggingFace Login failed. Models might not download if private. Error: {e}")

        # Define SAM3 Root to find assets (heuristic based on user snippet)
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")

        # Optimization settings
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        try:
            self.model = build_sam3_image_model(bpe_path=bpe_path)
        except Exception:
            print("Warning: Could not locate BPE file at derived path. Trying default build...")
            self.model = build_sam3_image_model()

        self.model.to(self.device)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        print("SAM3 Initialization Complete.")

    def detect_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Runs SAM3 inference on the image.
        Returns a list of binary masks (numpy arrays of shape (H, W)).
        """
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
                    # Note: inference_state is updated in place or returned
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state, 
                        prompt=self.text_prompt
                    )
                    
                    # 4. Extract Masks
                    # SAM3 'inference_state' usually contains the prediction results after prompting.
                    # We need to parse the specific structure of inference_state to get masks.
                    # Based on SAM3 internals: usually 'pred_masks' key holds the tensor.
                    
                    masks_tensor = None
                    if hasattr(inference_state, 'pred_masks'):
                         masks_tensor = inference_state.pred_masks
                    elif isinstance(inference_state, dict) and 'pred_masks' in inference_state:
                         masks_tensor = inference_state['pred_masks']
                    else:
                        # Fallback: inspect the object if structure is unknown (debugging)
                        # Assuming typical SAM output format: (Batch, N, H, W)
                        print("Warning: Could not find 'pred_masks' directly. Checking 'masks'...")
                        if 'masks' in inference_state:
                             masks_tensor = inference_state['masks']

                    if masks_tensor is None:
                        return []

                    # Process Masks: Tensor -> List of Numpy Arrays
                    # Masks are usually bool or float logits. We need uint8 binary (0, 1).
                    output_masks = []
                    
                    # Ensure we are on CPU and numpy
                    if isinstance(masks_tensor, torch.Tensor):
                        masks_np = masks_tensor.detach().cpu().numpy()
                    else:
                        masks_np = masks_tensor

                    # Iterate over detected objects
                    # Shape is likely (N_objects, 1, H, W) or (N_objects, H, W)
                    for m in masks_np:
                        # Remove channel dim if present (1, H, W) -> (H, W)
                        if m.ndim == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        
                        # Threshold if logits (unlikely for final output, but safety check)
                        # Usually SAM returns boolean masks or 0.0-1.0 probabilities
                        if m.dtype != bool and m.max() > 1.0: 
                            binary_mask = (m > 0).astype(np.uint8) # Logits
                        else:
                            binary_mask = m.astype(np.uint8)

                        # Resize if mask resolution differs from original image (SAM sometimes predicts at low res)
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