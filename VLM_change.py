import os
import glob
import time
import PIL.Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging

logger = logging.getLogger(__name__)

class VLMProcessor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.configured = False
        self._configure()
        self.model = None

    def _configure(self):
        if not self.api_key:
            logger.error("API Key missing for VLM.")
            return
        genai.configure(api_key=self.api_key)
        self.configured = True
        self.model_name = self._get_best_model()
        if self.model_name:
            self.model = genai.GenerativeModel(self.model_name)

    def _get_best_model(self):
        """Dynamically finds a working Vision-Language Model."""
        try:
            models = list(genai.list_models())
            model_names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            priorities = [
                "models/gemini-2.5-flash",
                "models/gemini-2.0-flash-exp",
                "models/gemini-1.5-flash", 
                "models/gemini-1.5-pro",
            ]

            for p in priorities:
                if p in model_names:
                    logger.info(f"VLM Selected: {p}")
                    return p
            
            # Fallback
            if model_names:
                return model_names[0]
            return None
        except Exception as e:
            logger.error(f"Error listing VLM models: {e}")
            return "models/gemini-1.5-flash" # Blind fallback

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
            return "Error: VLM Model not initialized."

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
            - **Note:** Images are crops that may have slight alignment artifacts. Ignore warp distortions.

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
            # prompt = """
            # ### Role
            # You are a forensic image analyst. Compare Image 1 (Reference) to Image 2 (Target) to detect physical modifications to the signage.
        
            # ### Context
            # - **Image 1:** Original Reference.
            # - **Image 2:** Current State.
            # - **Wrap Warning:** Images are digitally warped. You MUST IGNORE "stretching," "skew," or "blur" caused by the warp process.
            # - **Segmentation Warning:** Images are crop from segmented part, so uou MUST take into account that sometimes the sign detection didn't crop them exactly the same way. Or may have bad segmentation.
        
            # ### Detection Checklist
            # Evaluate the following 4 distinct categories. For each, determine if a **PHYSICAL** change occurred.
        
            # 1.  **Text Content:**
            #     - Check for missing letters, added graffiti text, or changed numbers.
            #     - Ignore text sharpness or pixelation.
        
            # 2.  **Shape & Structure:**
            #     - Look for: Dents, bent metal, broken corners, or holes.
            #     - **CRITICAL:** Do NOT report perspective distortion or trapezoidal shapes and normal signs of aging (weathering, scratches) as a change. ONLY report physical deformation of the material. 
            
            # 3.  **Logo & Iconography:**
            #     - Check if the logo/symbol is missing, covered (sticker/graffiti), or scraped off.
            #     - Check if the icon type has changed (e.g., "Left Arrow" became "Right Arrow").
        
            # 4.  **Color:**
            #     - Look for: Complete repainting (e.g., Red sign became Blue) or severe bleaching.
            #     - **CRITICAL:** Do NOT report "darker" or "lighter" tones caused by shadows, sun glare, camera white balance, color fainting, or different lighting conditions.
        
            # ### Output Format
            # Provide your analysis in this exact format:
        
            # **1. Summary:** [One sentence summary]
        
            # **2. Detailed Changes:**
            # * **Text:** [No Change / Change Detected: description]
            # * **Shape:** [No Change / Change Detected: description]
            # * **Logo:** [No Change / Change Detected: description]
            # * **Color:** [No Change / Change Detected: description]
        
            # **3. Severity:** [No-change | Low (Cosmetic) | Medium (Readable but damaged) | High (Meaning altered/destroyed)]
            # **4. Confidence:** [Sure | Doubt | Not-sure]
            # """

            generation_config = {
                "temperature": 0.1,  # Keep very low to force strict adherence to the "Ignore" rules
                "top_p": 0.8,
                "top_k": 40
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
