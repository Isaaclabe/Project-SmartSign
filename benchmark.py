import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    SamModel,
    SamProcessor,
    CLIPProcessor,
    CLIPModel
)
from PIL import Image
import sys
import torchvision.transforms as T

# Try imports for advanced methods
try:
    from diffusers import StableDiffusionPipeline
except ImportError:
    StableDiffusionPipeline = None

try:
    import timm
except ImportError:
    timm = None

# ----------------------------
# 1. Configuration & Utils
# ----------------------------
# Token provided in your snippet
# Read token from environment (never hard-code secrets in git)
HF_TOKEN = "your_huggingface_token_here"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def find_result_image(base_folder, face_id="face1"):
    """
    Locates the single result image in the output folder.
    Matches folder structure from previous pipeline: ./dataX/face1_signs/
    """
    # Try plural "signs" (from pipeline) or singular "sign" (from prompt text)
    possible_folders = [
        os.path.join(base_folder, f"{face_id}_signs"),
        os.path.join(base_folder, f"{face_id}_sign")
    ]
    
    target_folder = None
    for f in possible_folders:
        if os.path.exists(f):
            target_folder = f
            break
    
    if not target_folder:
        raise FileNotFoundError(f"Could not find result folder for {face_id} in {base_folder}")

    # Find valid images
    exts = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(target_folder, ext)))
    
    # Filter out 'mask_' images if present, prefer 'crop' or 'reference'
    # The user said "there is only one image", so we take the first one found.
    if not files:
        raise FileNotFoundError(f"No images found in {target_folder}")
        
    return files[0]

def load_image(path):
    img = cv2.imread(path)
    if img is None: raise ValueError(f"Cannot load {path}")
    return img

def to_tensor(img, processor, device):
    return processor(images=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), return_tensors="pt").to(device)

def cosine_similarity_map(feat1, feat2, shape=None):
    """Computes cosine similarity map between two feature grids."""
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)

    # Cosine Similarity
    sim = (feat1 * feat2).sum(dim=-1).detach().cpu().numpy()

    # Reshape to grid
    if shape is not None:
        try:
            sim_map = sim.reshape(shape[0], shape[1])
        except ValueError:
             side = int(np.sqrt(sim.shape[0]))
             sim_map = sim[:side*side].reshape(side, side)
    else:
        side = int(np.sqrt(sim.shape[0]))
        if side * side != sim.shape[0]:
            sim_map = sim.reshape(side, -1)
        else:
            sim_map = sim.reshape(side, side)

    return 1.0 - sim_map # Distance (0=Same, 1=Diff)

# ----------------------------
# 2. SCORER Implementation (Non-Parametric)
# ----------------------------
class SCORER_NonParametric_Reconstruction(torch.nn.Module):
    """
    Non-Parametric Cross-Attention for SCORER (No Training Needed).
    Reconstructs 'Query' (Tgt) using 'Key/Value' (Ref) based on similarity.
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        # query: [1, N_tgt, Dim]
        # key:   [1, N_ref, Dim]
        # value: [1, N_ref, Dim]

        # 1. Calculate Attention Scores (Similarity between Tgt and Ref patches)
        # Scale by sqrt(dim) for stability
        scale = query.shape[-1] ** -0.5
        attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_logits, dim=-1) # [1, N_tgt, N_ref]

        # 2. Reconstruct Tgt using weighted sum of Ref patches
        reconstruction = torch.matmul(attn_weights, value) # [1, N_tgt, Dim]

        return reconstruction

# ----------------------------
# 3. Advanced Methods Classes
# ----------------------------

class MethodDreamSim:
    def __init__(self):
        try:
            from dreamsim import dreamsim
            model, preprocess = dreamsim(pretrained=True)
            self.model = model.to(DEVICE).eval()
            print("   [DreamSim] Model Loaded.")
        except ImportError:
            print("   [Warning] DreamSim not installed. (pip install git+https://github.com/ssundaram21/dreamsim.git)")
            self.model = None
        except Exception as e:
            print(f"   [Error] DreamSim load failed: {e}")
            self.model = None

    def run(self, img_ref_pil, img_tgt_pil):
        if self.model is None: return np.zeros((64,64))

        stride = 64
        window = 64
        t = T.ToTensor()
        ref_t = t(img_ref_pil).to(DEVICE)
        tgt_t = t(img_tgt_pil).to(DEVICE)

        def get_embeddings(tensor_img):
            h, w = tensor_img.shape[1], tensor_img.shape[2]
            patches = []
            for y in range(0, h - window + 1, stride):
                for x in range(0, w - window + 1, stride):
                    crop = tensor_img[:, y:y+window, x:x+window]
                    patches.append(crop)
            if not patches: return None, 0, 0

            feats = []
            batch_size = 32
            with torch.no_grad():
                for i in range(0, len(patches), batch_size):
                    batch = torch.stack(patches[i:i+batch_size])
                    batch = F.interpolate(batch, size=(224, 224), mode='bilinear')
                    embed = self.model.embed(batch)
                    feats.append(F.normalize(embed, dim=-1))

            grid_h = (h - window) // stride + 1
            grid_w = (w - window) // stride + 1
            return torch.cat(feats, dim=0), grid_h, grid_w

        ref_feats, rh, rw = get_embeddings(ref_t)
        tgt_feats, th, tw = get_embeddings(tgt_t)

        if ref_feats is None: return np.zeros((64,64))

        sim_matrix = torch.mm(tgt_feats, ref_feats.T)
        v_tgt, _ = sim_matrix.max(dim=1)
        map_tgt = (1.0 - v_tgt.cpu().numpy()).reshape(th, tw)

        return map_tgt

class MethodDiffSim:
    def __init__(self):
        if StableDiffusionPipeline is None:
            print("   [Warning] Diffusers not installed. (pip install diffusers)")
            self.pipe = None
            return
        try:
            print("   [DiffSim] Loading Stable Diffusion 1.5...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to(DEVICE)
            self.pipe.vae.to(torch.float32)
        except Exception as e:
            print(f"   [Error] Could not load DiffSim (SD): {e}")
            self.pipe = None

    def run(self, img_ref_pil, img_tgt_pil):
        if self.pipe is None: return np.zeros((64,64))

        timestep = 100

        def extract_sd_features(img_pil):
            img = img_pil.resize((512, 512))
            img_t = T.ToTensor()(img).unsqueeze(0).to(DEVICE) * 2.0 - 1.0

            with torch.no_grad():
                latents = self.pipe.vae.encode(img_t.to(torch.float32)).latent_dist.sample() * 0.18215
                latents = latents.to(self.pipe.unet.dtype)

                noise = torch.randn_like(latents)
                t = torch.tensor([timestep], device=DEVICE).long()
                noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)

                features = []
                def hook_fn(module, input, output):
                    features.append(output)

                handle = self.pipe.unet.up_blocks[1].register_forward_hook(hook_fn)

                encoder_hidden_states = torch.zeros((1, 77, 768), device=DEVICE, dtype=latents.dtype)
                self.pipe.unet(noisy_latents, t, encoder_hidden_states=encoder_hidden_states)
                handle.remove()

                feat_map = features[0]
                feat_map = F.normalize(feat_map, dim=1)

                b, c, h, w = feat_map.shape
                flat_feats = feat_map.view(c, -1).permute(1, 0)
                return flat_feats, h, w

        f_ref, rh, rw = extract_sd_features(img_ref_pil)
        f_tgt, th, tw = extract_sd_features(img_tgt_pil)

        sim_matrix = torch.mm(f_tgt, f_ref.T)
        val_tgt, _ = sim_matrix.max(dim=1)
        map_tgt = 1.0 - val_tgt
        map_tgt = map_tgt.view(th, tw).float().cpu().numpy()

        return map_tgt

class MethodSCORER:
    def __init__(self):
        if timm is None:
            print("   [Warning] Timm not installed.")
            self.backbone = None
            return

        print(f"   [SCORER] Loading DINOv2 Backbone...")
        try:
            self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, dynamic_img_size=True).to(DEVICE).eval()
            self.reconstructor = SCORER_NonParametric_Reconstruction() # Use non-parametric fixed version

            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"   [Error] SCORER load failed: {e}")
            self.backbone = None

    def run(self, pil_ref, pil_tgt):
        if self.backbone is None: return np.zeros((64,64))

        # Resize to fixed grid multiple
        size = (644, 644)
        ref_t = self.transform(pil_ref.resize(size)).unsqueeze(0).to(DEVICE)
        tgt_t = self.transform(pil_tgt.resize(size)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            f_ref = self.backbone.forward_features(ref_t)[:, 1:, :]
            f_tgt = self.backbone.forward_features(tgt_t)[:, 1:, :]

            f_ref = F.normalize(f_ref, dim=-1)
            f_tgt = F.normalize(f_tgt, dim=-1)

            # Reconstruction: Reconstruct Tgt (Query) using Ref (Key/Value)
            # "Can we explain Tgt using pieces of Ref?"
            f_tgt_recon = self.reconstructor(f_tgt, f_ref, f_ref)

            # If Tgt has new object, reconstruction will fail (high difference)
            similarity = F.cosine_similarity(f_tgt, f_tgt_recon, dim=-1)
            change_score = 1.0 - similarity

            h_grid, w_grid = size[1] // 14, size[0] // 14
            heatmap = change_score.view(h_grid, w_grid).cpu().numpy()

        return heatmap

# ----------------------------
# 4. The Tracker (Alignment Engine)
# ----------------------------
class ImageTracker:
    def __init__(self):
        self.loftr = None

    def align_sift(self, img_ref, img_tgt):
        gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
        gray_tgt = cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_ref, None)
        kp2, des2 = sift.detectAndCompute(gray_tgt, None)

        if des1 is None or des2 is None: return None

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good.append(m)

        if len(good) < 4: return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None: return None

        h, w = img_ref.shape[:2]
        return cv2.warpPerspective(img_tgt, M, (w, h))

    def align_loftr(self, img_ref, img_tgt):
        try:
            from kornia.feature import LoFTR
            if self.loftr is None:
                print("   Initializing LoFTR for tracking...")
                self.loftr = LoFTR(pretrained="outdoor").to(DEVICE)

            img0 = torch.from_numpy(cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)).float()[None, None].to(DEVICE) / 255.0
            img1 = torch.from_numpy(cv2.cvtColor(img_tgt, cv2.COLOR_BGR2GRAY)).float()[None, None].to(DEVICE) / 255.0

            with torch.no_grad():
                correspondences = self.loftr({'image0': img0, 'image1': img1})
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()

            if len(mkpts0) < 4: return None
            M, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
            if M is None: return None
            h, w = img_ref.shape[:2]
            return cv2.warpPerspective(img_tgt, M, (w, h))
        except Exception:
            return self.align_sift(img_ref, img_tgt)

# ----------------------------
# 5. Change Detector
# ----------------------------
class ChangeDetector:
    def __init__(self):
        self.tracker = ImageTracker()
        self.cache = {}
        self.dreamsim_model = None
        self.diffsim_model = None
        self.scorer_model = None

    def get_aligned(self, img_ref, img_tgt, method="sift"):
        key = f"{method}_{img_ref.shape}_{img_tgt.shape}"
        if key in self.cache: return self.cache[key]

        print(f"   Aligning images using {method}...")
        if method == "sift": aligned = self.tracker.align_sift(img_ref, img_tgt)
        elif method == "loftr": aligned = self.tracker.align_loftr(img_ref, img_tgt)
        else: aligned = None

        if aligned is None:
            print("   Alignment Failed! Using raw resize.")
            aligned = cv2.resize(img_tgt, (img_ref.shape[1], img_ref.shape[0]))

        self.cache[key] = aligned
        return aligned

    def run_method(self, name, img_ref, img_tgt, tracking=False):
        print(f"Running Method: {name}...")

        align_method = "loftr" if tracking else "sift"
        img_aligned = self.get_aligned(img_ref, img_tgt, align_method)

        pil_ref = Image.fromarray(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
        pil_aligned = Image.fromarray(cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB))

        mask = None

        if "DreamSim" in name:
            if not self.dreamsim_model: self.dreamsim_model = MethodDreamSim()
            mask = self.dreamsim_model.run(pil_ref, pil_aligned)

        elif "DiffSim" in name:
            if not self.diffsim_model: self.diffsim_model = MethodDiffSim()
            mask = self.diffsim_model.run(pil_ref, pil_aligned)

        elif "SCORER" in name:
            if not self.scorer_model: self.scorer_model = MethodSCORER()
            mask = self.scorer_model.run(pil_ref, pil_aligned)

        elif "DINOv2" in name:
            mask = self._deep_feature_diff(img_ref, img_aligned, "facebook/dinov2-small")
        elif "DINOv3" in name:
            mask = self._deep_feature_diff(img_ref, img_aligned, "facebook/dinov3-vitb16-pretrain-lvd1689m")
        elif "OpenCLIP" in name:
            mask = self._clip_diff(img_ref, img_aligned)
        elif "SAM" in name:
            mask = self._sam_diff(img_ref, img_aligned)
        elif "DepthAnything" in name:
            mask = self._depth_diff(img_ref, img_aligned)
        else:
            mask = self._pixel_diff(img_ref, img_aligned)

        if mask is not None:
            mask = cv2.resize(mask, (img_ref.shape[1], img_ref.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = np.clip(mask, 0, 1)
        return mask

    def _pixel_diff(self, img1, img2):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2) / 255.0
        return diff

    def _deep_feature_diff(self, img1, img2, model_id):
        try:
            if "dinov3" in model_id:
                # DINOv3 might need token or specific loading if gated
                processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", token=HF_TOKEN)
                model = AutoModel.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True).to(DEVICE)
            else:
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = AutoModel.from_pretrained(model_id).to(DEVICE)

            inputs1 = to_tensor(img1, processor, DEVICE)
            inputs2 = to_tensor(img2, processor, DEVICE)

            patch_size = 14
            h, w = inputs1.pixel_values.shape[-2:]
            h_grid, w_grid = h // patch_size, w // patch_size
            num_spatial = h_grid * w_grid

            with torch.no_grad():
                out1 = model(**inputs1).last_hidden_state
                out2 = model(**inputs2).last_hidden_state
                feat1 = out1[:, -num_spatial:, :]
                feat2 = out2[:, -num_spatial:, :]

            return cosine_similarity_map(feat1[0], feat2[0], shape=(h_grid, w_grid))
        except Exception as e:
            print(f"   [!] Error {model_id}: {e}")
            return np.zeros((64,64))

    def _clip_diff(self, img1, img2):
        try:
            model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
            processor = CLIPProcessor.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id).to(DEVICE)

            inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.vision_model(pixel_values=inputs.pixel_values).last_hidden_state
                feat1, feat2 = out[0, 1:], out[1, 1:]
            return cosine_similarity_map(feat1, feat2, None)
        except Exception:
            return np.zeros((64,64))

    def _sam_diff(self, img1, img2):
        try:
            model_id = "facebook/sam-vit-base"
            processor = SamProcessor.from_pretrained(model_id)
            model = SamModel.from_pretrained(model_id).to(DEVICE)

            h_orig, w_orig = img1.shape[:2]

            inputs = processor(images=[img1, img2], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                embed = model.get_image_embeddings(inputs.pixel_values)
                scale = 1024 / max(h_orig, w_orig)
                new_h = int(h_orig * scale + 0.5)
                # SAM embeddings are 64x64 for 1024x1024
                valid_h_feat = int(new_h / 1024 * 64)
                valid_w_feat = int(int(w_orig * scale + 0.5) / 1024 * 64)

                feat1 = embed[0, :, :valid_h_feat, :valid_w_feat]
                feat2 = embed[1, :, :valid_h_feat, :valid_w_feat]

                c = feat1.shape[0]
                feat1 = feat1.permute(1, 2, 0).reshape(-1, c)
                feat2 = feat2.permute(1, 2, 0).reshape(-1, c)

            return cosine_similarity_map(feat1, feat2, shape=(valid_h_feat, valid_w_feat))
        except Exception as e:
            print(f"SAM Error: {e}")
            return np.zeros((64,64))

    def _depth_diff(self, img1, img2):
        try:
            model_id = "LiheYoung/depth-anything-small-hf"
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = AutoModelForDepthEstimation.from_pretrained(model_id).to(DEVICE)

            inputs1 = to_tensor(img1, processor, DEVICE)
            inputs2 = to_tensor(img2, processor, DEVICE)

            with torch.no_grad():
                d1 = model(**inputs1).predicted_depth
                d2 = model(**inputs2).predicted_depth

            d1 = F.interpolate(d1.unsqueeze(1), size=img1.shape[:2], mode="bicubic").squeeze()
            d2 = F.interpolate(d2.unsqueeze(1), size=img1.shape[:2], mode="bicubic").squeeze()
            diff = torch.abs(d1 - d2).cpu().numpy()
            return (diff - diff.min()) / (diff.max() - diff.min() + 1e-6)
        except Exception:
            return np.zeros((64,64))

# ----------------------------
# 6. Execution & Interpretation
# ----------------------------
def run_benchmark():
    detector = ChangeDetector()
    
    print("\n[INIT] Locating input images...")
    try:
        path_ref = find_result_image("./data1", face_id="face1")
        print(f"   Reference (data1): {path_ref}")
        
        path_tgt = find_result_image("./data2", face_id="face1")
        print(f"   Target (data2):    {path_tgt}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("Please ensure you have run the 'main_pipeline.py' first.")
        return

    img_ref = load_image(path_ref)
    img_tgt = load_image(path_tgt)

    methods = [
        "DINOv2", "DINOv3", "SAM", "OpenCLIP",
        "DreamSim", "DiffSim", "SCORER",
        "DepthAnything",
        "LoFTR (Tracking Only)",
        "DINOv3 + tracking"
    ]

    results = []
    for m_name in methods:
        tracking = "tracking" in m_name or "LoFTR" in m_name
        res = detector.run_method(m_name, img_ref, img_tgt, tracking=tracking)
        results.append((m_name, res))

    # Visualization
    cols = 4
    rows = (len(methods) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f"Comprehensive Change Detection Heatmaps\nRef vs Tgt", fontsize=16)

    # Interpretation Guide
    print("\n" + "="*50)
    print("HOW TO READ THESE HEATMAPS")
    print("="*50)
    print("Each image shows a 'Heatmap of Change' where:")
    print(" - BLUE / DARK Areas: NO Change (High Similarity)")
    print(" - RED / BRIGHT Areas: HIGH Change (Low Similarity)")
    print("\nINTERPRETATION BY METHOD:")
    print("1. DINOv2 / SCORER: Highlights Semantic Changes (new objects, missing parts). Ignores lighting.")
    print("2. DreamSim / DiffSim: Perceptual Changes. Similar to human vision.")
    print("3. DepthAnything: Highlights geometry changes (e.g. object moved closer/further).")
    print("4. LoFTR / Tracking: Shows misalignment residuals.")
    print("="*50 + "\n")

    axes = axes.flatten()
    for i, (name, res) in enumerate(results):
        im = axes[i].imshow(res, cmap='jet', vmin=0, vmax=1)
        axes[i].set_title(name)
        axes[i].axis('off')

    for i in range(len(results), len(axes)): axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_benchmark()
