import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from collections import Counter
import openslide
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from pathlib import Path
import h5py

# Optional: show all output in a Jupyter Notebook cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# =====================================================================================
# 1. INITIAL SETUP AND CONFIGURATION
# =====================================================================================
print("--- Initializing Model and Settings ---")

# --- Model Configuration ---
model_cfg = 'conch_ViT-B-16'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/data/wentao/CONCH/model/pytorch_model.bin'

# --- GLOBAL LABEL & COLOR CONFIGURATION ---
CLASSES = [
    "malignant tissue", "benign tissue", "stroma", "lymphocytes",
    "necrosis", "adipose tissue", "tissue artifact", "blood vessel",
    "extracellular mucin", "nerve", "hemorrhage", "smooth muscle", "plasma cells"
]

LABEL_TO_COLOR = {
    "malignant tissue": (217, 30, 30), "benign tissue": (60, 133, 194),
    "stroma": (250, 194, 99), "lymphocytes": (90, 186, 125),
    "necrosis": (128, 0, 128), "adipose tissue": (245, 237, 203),
    "tissue artifact": (128, 128, 128), "blood vessel": (255, 182, 193),
    "extracellular mucin": (0, 255, 255), "nerve": (75, 0, 130),
    "hemorrhage": (165, 42, 42), "smooth muscle": (210, 105, 30),
    "plasma cells": (255, 20, 147)
}

# 13 Prompt templates for robust Zero-Shot classification
TEMPLATES = [
    "an H&E stained image of CLASSNAME.", "a photomicrograph showing CLASSNAME.",
    "tissue section showing CLASSNAME.", "area dominated by CLASSNAME.",
    "patch with abundant CLASSNAME.", "region with evidence of CLASSNAME.",
    "an example of CLASSNAME.", "this is CLASSNAME.", "presence of CLASSNAME.",
    "CLASSNAME is present.", "an image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.", "a histopathological image of CLASSNAME."
]

# --- Load CONCH Model and Preprocessor ---
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
_ = model.eval()

# --- Generate Text Embeddings via Prompt Ensembling ---
tokenizer = get_tokenizer()
prompts = [template.replace("CLASSNAME", class_name) for class_name in CLASSES for template in TEMPLATES]
tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
with torch.no_grad():
    all_text_embeddings = model.encode_text(tokenized_prompts)
    all_text_embeddings /= all_text_embeddings.norm(dim=-1, keepdim=True)

# Average embeddings across templates for each class
emb_per_class = len(TEMPLATES)
class_text_embeddings = []
for i in range(len(CLASSES)):
    class_embeds = all_text_embeddings[i*emb_per_class:(i+1)*emb_per_class]
    avg_embed = class_embeds.mean(dim=0)
    avg_embed /= avg_embed.norm()
    class_text_embeddings.append(avg_embed)
CLASS_TEXT_EMBEDDINGS = torch.stack(class_text_embeddings, dim=0)
print(f"Model and text embeddings for {len(CLASSES)} classes are ready.")


# =====================================================================================
# 2. HELPER FUNCTIONS
# =====================================================================================
def extract_coords(filename):
    """
    Parses (x, y) coordinates from PNG filenames.
    Assumes filename format: {x}_{y}.png
    """
    match = re.search(r'(\d+)_(\d+)\.png$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# =====================================================================================
# 3. MAIN PROCESSING FUNCTION
# =====================================================================================
def process_slide(slide_id: str, patch_dir: str, svs_path: str, original_h5_path: str,
                  output_dir: str, patch_size: int):
    
    print(f"\n{'-'*30}\nProcessing Slide ID: {slide_id}\n{'-'*30}")
    
    # Define file paths inside the slide-specific folder
    output_csv_path = os.path.join(output_dir, f"{slide_id}.csv")
    output_h5_path = os.path.join(output_dir, f"{slide_id}.h5")
    output_dist_path = os.path.join(output_dir, f"{slide_id}_dist.png")
    output_vis_path = os.path.join(output_dir, f"{slide_id}_vis.png")

    # --- STEP 1: Inference on Patches ---
    print(f"-> Step 1: Running inference on .png patches from {patch_dir}")
    image_paths = glob.glob(os.path.join(patch_dir, '**/*.png'), recursive=True)
    
    if not image_paths:
        print(f"WARNING: No .png patches found in '{patch_dir}'. Skipping.")
        return
        
    results = []
    for img_path in tqdm(image_paths, desc="Inference"):
        image = Image.open(img_path).convert('RGB').resize((224, 224))
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(image_tensor)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            # Similarity score calculation
            logits = (image_embedding @ CLASS_TEXT_EMBEDDINGS.T) * model.logit_scale.exp()
            sim_scores = logits.softmax(dim=-1).cpu().numpy()[0]
        
        pred_idx = sim_scores.argmax()
        results.append({'file': img_path, 'pred_class': CLASSES[pred_idx]})
        
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

    # --- STEP 2: Prediction Distribution Plot ---
    counter = Counter(df['pred_class'])
    plt.figure(figsize=(12, 7))
    plt.bar(counter.keys(), counter.values())
    plt.title(f'Prediction Distribution for {slide_id}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dist_path)
    plt.close()

    # --- STEP 3: Weak-Label H5 Generation ---
    # Map coordinates to predicted labels
    pred_map = {extract_coords(os.path.basename(r['file'])): r['pred_class'] for _, r in df.iterrows()}
    
    try:
        with h5py.File(original_h5_path, 'r') as f_in:
            original_coords = f_in['coords'][:]
            # Map labels (assumes patch filename coordinates are level-0-coords // 2)
            weak_labels = [pred_map.get((c[0]//2, c[1]//2), 'Unknown') for c in original_coords]
        
        with h5py.File(output_h5_path, 'w') as f_out:
            f_out.create_dataset('coords', data=original_coords)
            f_out.create_dataset('labels', data=np.array(weak_labels, dtype=h5py.string_dtype()))
    except Exception as e:
        print(f"ERROR: H5 generation failed: {e}")

    # --- STEP 4: Tissue Segmentation Map Visualization ---
    print(f"-> Step 4: Generating Visualization for {slide_id}")
    slide = openslide.OpenSlide(svs_path)
    # Create blank canvas sized to the WSI dimensions
    canvas = Image.new('RGB', slide.dimensions, (255, 255, 255))
    
    for _, row in df.iterrows():
        x, y = extract_coords(os.path.basename(row['file']))
        if x is not None:
            color = LABEL_TO_COLOR.get(row['pred_class'], (255, 255, 255))
            # Calculate physical position on canvas (Index * Patch Size)
            paste_pos = (x * patch_size, y * patch_size)
            canvas.paste(Image.new('RGB', (patch_size, patch_size), color), paste_pos)
            
    # Resize reconstruction and original thumbnail for side-by-side display
    thumb = slide.get_thumbnail((slide.dimensions[0] // 32, slide.dimensions[1] // 32))
    recon_thumb = canvas.resize(thumb.size, Image.Resampling.BICUBIC)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    
    ax1.imshow(thumb)
    ax1.set_title("Original Slide (Thumbnail)", fontsize=14)
    ax1.axis('off')
    
    ax2.imshow(recon_thumb)
    # Subplot title including Slide ID
    ax2.set_title("Segmentation Map: " + slide_id, fontsize=14)
    ax2.axis('off')
    
    # Add legend
    legend_patches = [mpatches.Patch(color=np.array(c)/255.0, label=l) for l, c in LABEL_TO_COLOR.items()]
    ax2.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1.02, 0.5), 
               title='Tissue Types', title_fontsize='large', fontsize='medium')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1.0]) 
    plt.savefig(output_vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"--- Finished processing {slide_id} ---")
    print(f"--- Results saved in: {output_dir} ---")


# =====================================================================================
# 4. EXECUTION
# =====================================================================================
if __name__ == '__main__':
    # Define Target Slide ID
    SLIDE_ID = "TCGA-2H-A9GQ-01Z-00-DX1"
    
    # Input Source Paths
    SVS_PATH = f"./WSI/{SLIDE_ID}.svs"
    PATCH_DIR = f"./Output/single_b20_t15/Patch/{SLIDE_ID}"
    ORIGINAL_H5_PATH = f"./Feature/h5_files/{SLIDE_ID}.h5"
    
    # Slide-specific Output Directory
    OUTPUT_BASE_DIR = f"./Segmentation/{SLIDE_ID}"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # Run the pipeline
    process_slide(
        slide_id=SLIDE_ID,
        patch_dir=PATCH_DIR,
        svs_path=SVS_PATH,
        original_h5_path=ORIGINAL_H5_PATH,
        output_dir=OUTPUT_BASE_DIR,
        patch_size=448
    )