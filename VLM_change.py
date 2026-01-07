import os
import glob
import time
import PIL.Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- CONFIGURATION ---
# PASTE YOUR KEY HERE (keep the quotes)
# Get one for free at: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY = "your_google_api_key_here" 

def configure_genai():
    """Configures the Gemini API using Env Var or Hardcoded Key."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Fallback to hardcoded key if env var is missing
    if not api_key:
        if GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY_HERE":
            api_key = GOOGLE_API_KEY
    
    if not api_key:
        print("\n" + "="*50)
        print("CRITICAL ERROR: API Key Missing")
        print("="*50)
        print("1. Open this file (vlm_analysis.py).")
        print("2. Find the line: GOOGLE_API_KEY = \"YOUR_GOOGLE_API_KEY_HERE\"")
        print("3. Replace the text inside quotes with your actual API key.")
        print("="*50 + "\n")
        return False
        
    genai.configure(api_key=api_key)
    return True

def get_best_model():
    """
    Dynamically finds a working Vision-Language Model (VLM).
    Prioritizes Flash (Nano/Fast) -> Pro -> Legacy Vision.
    """
    print("Checking available models...")
    try:
        models = list(genai.list_models())
        model_names = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        
        # Priority list
        priorities = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-latest",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-pro",
            "models/gemini-pro-vision"
        ]
        
        for p in priorities:
            if p in model_names:
                print(f"   -> Selected Model: {p}")
                return p
        
        # Fallback
        for m in model_names:
            if 'flash' in m or 'vision' in m:
                print(f"   -> Selected Fallback Model: {m}")
                return m

        print("ERROR: No suitable VLM model found in your account.")
        return None

    except Exception as e:
        print(f"Error listing models: {e}")
        return "models/gemini-1.5-flash"

def find_images(folder, face_id="face1"):
    """Locates the crop/result image in the data folders."""
    if not os.path.exists(folder):
        print(f"Warning: Folder {folder} does not exist.")
        return None

    search_paths = [
        os.path.join(folder, f"{face_id}_signs"),
        os.path.join(folder, f"{face_id}_sign")
    ]
    for p in search_paths:
        if os.path.exists(p):
            files = glob.glob(os.path.join(p, "*.png")) + glob.glob(os.path.join(p, "*.jpg"))
            # Sort to ensure deterministic selection if multiple exist
            files.sort()
            if files: return files[0]
    return None

def analyze_changes_with_vlm(ref_path, tgt_path):
    model_name = get_best_model()
    if not model_name:
        return "Could not initialize model."

    print(f"Initializing {model_name}...")
    model = genai.GenerativeModel(model_name)

    print("Uploading images...")
    img_ref = PIL.Image.open(ref_path)
    img_tgt = PIL.Image.open(tgt_path)

    prompt = """
    You are a forensic image analyst. 
    Image 1 is the REFERENCE (original state).
    Image 2 is the TARGET (current state).

    Compare them and identify specific changes in the sign. Focus on:
    1. Text content differences.
    2. Shape change.
    3. Texture or logo change.
    4. Color change (not color fading).

    and assume that there are not both esspecially taken from the same point of view (but there are wrapped to appear), 
    and not the same lighting conditions.

    Output format:
    - Summary: [One sentence summary]
    - Detected Changes: [List of bullet points]
    - Severity: [No-change/Low/Medium/High]
    - Coonfidence [Sure/Doubt/Not-sure]
    """

    print("Sending request to Google API...")
    try:
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            [prompt, img_ref, img_tgt],
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        return f"API Error: {e}"

if __name__ == "__main__":
    if not configure_genai():
        exit()

    # 1. Locate Images
    print("Locating images in ./data1 and ./data2...")
    path_ref = find_images("/content/Project-SmartSign/data1", "face1")
    path_tgt = find_images("/content/Project-SmartSign/data2", "face1")

    if not path_ref or not path_tgt:
        print(f"Error: Could not find images.\nRef: {path_ref}\nTgt: {path_tgt}")
        print("Make sure you ran the main_pipeline.py first.")
    else:
        print(f"Reference: {path_ref}")
        print(f"Target:    {path_tgt}")
        
        # Check for user error (comparing same image)
        if os.path.abspath(path_ref) == os.path.abspath(path_tgt):
            print("\nWARNING: Reference and Target paths are IDENTICAL.")
            print("The model will not find any changes because you are comparing the image to itself.")
            print("Check that ./data2 contains different images than ./data1.\n")
        
        print("-" * 40)
        
        # 2. Run Analysis
        analysis = analyze_changes_with_vlm(path_ref, path_tgt)
        
        print("\n--- VLM FORENSIC REPORT ---")
        print(analysis)
        print("---------------------------")