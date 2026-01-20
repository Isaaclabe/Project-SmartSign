import os
import glob
import time
import PIL.Image
import warnings
import logging

# Suppress the specific Google Generative AI deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

class VLMProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        self._configure()

    def _configure(self):
        if not self.api_key:
            logger.error("API Key missing for VLM.")
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model_name = self._get_best_model()
            
            if self.model_name:
                logger.info(f"Initializing VLM with model: {self.model_name}")
                self.model = genai.GenerativeModel(self.model_name)
            else:
                logger.error("No suitable VLM model found.")
                # Force fallback attempt anyway
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                
        except Exception as e:
            logger.error(f"VLM Configuration Error: {e}")
            # Try blind fallback
            try:
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except:
                pass

    def _get_best_model(self):
        """Dynamically finds a working Vision-Language Model."""
        try:
            # Try to list models. If this fails (e.g. auth error), we catch exception.
            models = list(genai.list_models())
            model_names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            priorities = [
                "models/gemini-3-flash-preview"
                "models/gemini-2.5-flash",
                "models/gemini-2.0-flash-exp",
                "models/gemini-1.5-flash", 
                "models/gemini-1.5-pro",
            ]

            for p in priorities:
                if p in model_names:
                    return p
            
            # Fallback to first available if priority list fails
            if model_names:
                return model_names[0]
                
        except Exception as e:
            logger.warning(f"Could not list models via API (likely network or auth issue). Defaulting to 'gemini-1.5-flash'. Error: {e}")
        
        # Hard fallback if listing fails
        return "models/gemini-1.5-flash"

    def resize_image_smart(self, pil_image, max_side=1024):
        width, height = pil_image.size
        if max(width, height) <= max_side:
            return pil_image
        ratio = max_side / max(width, height)
        return pil_image.resize((int(width * ratio), int(height * ratio)), PIL.Image.Resampling.LANCZOS)

    def analyze_pair(self, ref_path: str, tgt_path: str) -> str:
        """
        Analyzes a pair of images (reference and target) for physical changes.
        """
        if not self.model:
            return "Error: VLM Model not initialized (Check API Key or Network)."

        try:
            img_ref = PIL.Image.open(ref_path)
            img_tgt = PIL.Image.open(tgt_path)

            img_ref = self.resize_image_smart(img_ref)
            img_tgt = self.resize_image_smart(img_tgt)

            prompt = """
            ### Role
            You are a forensic image analyst. Compare Image 1 (Reference) to Image 2 (Target) to detect physical modifications to the signage.

            ### Context
            - **Image 1:** Original Reference.
            - **Image 2:** Current State.
            - **Note:** Images are crops. Ignore slight misalignment or warp artifacts.

            ### Detection Checklist
            Evaluate for **PHYSICAL** changes only:
            1.  **Text Content:** Missing/Added text, changed numbers, graffiti.
            2.  **Shape & Structure:** Dents, bends, broken parts.
            3.  **Logo/Icons:** Missing, covered, or altered symbols.
            4.  **Color:** Repainting or severe bleaching (Ignore lighting/shadows).

            ### Output Format
            Provide your analysis in this exact format:
            **1. Summary:** [One sentence]
            **2. Changes:**
            * [Change 1 or "None"]
            * [Change 2 or "None"]
            **3. Severity:** [No-change | Low | Medium | High]
            **4. Confidence:** [Sure | Doubt]
            """
            
            generation_config = {
                "temperature": 0.1,  # Keep very low to force strict adherence to the "Ignore" rules
                "top_p": 0.8,
                "top_k": 40
                }
            
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            response = self.model.generate_content(
                [prompt, img_ref, img_tgt],
                #safety_settings=safety_settings
                generation_config=generation_config
            )
            
            return response.text

        except Exception as e:
            logger.error(f"VLM Analysis Failed: {e}")
            return f"Analysis Error: {e}"
