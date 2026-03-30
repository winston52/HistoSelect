# HistoSelect
* This repository contains the official implementation for our work **"Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning"**, accepted by **CVPR 2026**.

![framework](./figure/HistoSelect_framework.png)

### Recent Updates
* **2026/03/30**: The preprocessing code for tiling WSI into patches.
* We are currently organizing the codebase. Stay tuned for further updates!

### Data Preparation
#### Step 1: Cut whole slide image into patches
```bash
python deepzoom_tiler_new.py \
    --slide_path /path/to/your/wsi_folder \
    --output_base /path/to/output_directory \
    -m 1 -b 40 -s 224 -j 32 -t 15 -o 40 -c True
```

#### Step 2: Extract the patch features