# WasteSegCluster: AI-Powered Waste Segmentation and Classification ♻️

## Overview
WasteSegCluster has presented an end to end computer vision system that has enabled machines to understand waste images in a practical and clear way. The system has been designed to detect waste objects in an image and outline each object precisely through segmentation and classify each item into the correct waste category. By integrating one segmentation workflow with two strong image classification workflows into a single unified pipeline this project has created a coherent approach using the WasteSegCluster dataset making it useful for both learning and real world waste management applications.

---

## Simple Explanation of What This Project Does
Imagine taking a photo of mixed garbage. This project helps AI answer:

1. **Where is each waste item?**
2. **What type of waste is it?**
3. **How well can the model separate similar-looking waste categories?**

This is useful for smart recycling systems, cleaner cities, and faster waste sorting support.

---

## Problem Statement
Waste sorting is still difficult in many real-world situations because:

The WasteSegCluster dataset was collected from several canals, lakes, and ponds to reflect real environmental conditions.

- Waste items overlap or are partially visible
- Different materials can look visually similar
- Manual sorting is slow, expensive, and inconsistent

This project addresses that by building a vision pipeline for **waste detection, segmentation, and classification** so waste can be identified more accurately and efficiently.

---

## Solution Approach
The project follows one coherent approach:

- Use **WasteSegCluster** (COCO-format segmentation data) as the core dataset
- Clean and rebalance data with image-level split control to reduce data leakage
- Train a segmentation model (**Mask R-CNN**) to locate and outline waste objects
- Extract object-focused regions from segmentation masks
- Train image classifiers (**TF-EfficientNetV2-M** and **ViT-Base-Patch16-224**) on those object regions
- Evaluate with visual and numeric metrics (AP/mAP, confusion matrix, classification report, accuracy)

This gives both:
- **Pixel-level understanding** (segmentation), and
- **Category-level recognition** (classification)

---

## Features / Highlights ✨
- End-to-end pipeline: data preparation → segmentation → region extraction → classification
- Uses **image-level stratified split (70/20/10)** to minimize leakage across train/validation/test
- Supports publication-quality visualizations and analysis plots
- Includes robust evaluation outputs:
  - COCO-style AP/mAP for segmentation
  - Confusion matrices
  - Classification reports
  - Accuracy and class-balance analysis
- Built with beginner-readable notebook flow and reproducible setup patterns

---

## Technologies Used 🧰

### Core
- Python
- Jupyter Notebook
- PyTorch

### Models and CV
- Detectron2 (Mask R-CNN)
- timm (EfficientNetV2, ViT)
- OpenCV
- Pillow (PIL)

### Data and Evaluation
- Roboflow
- NumPy
- pandas
- scikit-learn
- pycocotools
- tqdm

### Visualization
- Matplotlib
- Seaborn

---

## Dataset Description (WasteSegCluster) 📦
**WasteSegCluster** is the central dataset used by this project.  
It is organized in COCO segmentation format with train/validation/test annotation files.

### What it contains
- Waste images
- Pixel-level polygon masks for object segmentation
- Category labels for waste types
- COCO JSON annotations

### Main waste classes used
- Composite_Rubber_Textile
- Organic
- Paper
- Plastic

A noisy/junk label is removed during preprocessing to improve training quality.

### Why this dataset is important
- It supports both **segmentation** and **classification** tasks
- It allows realistic waste-scene modeling (not just clean studio images)
- It helps build practical AI for smart waste management and recycling automation

---

## Workflow / Methodology 🔄
1. Download WasteSegCluster in COCO segmentation format.
2. Inspect annotations and verify segmentation quality.
3. Remove unwanted/noisy class entries.
4. Merge available splits and perform image-level stratified rebalance.
5. Save clean train/val/test partitions (70/20/10).
6. Train Mask R-CNN for waste instance segmentation.
7. Evaluate segmentation performance (COCO evaluator, AP/mAP).
8. Extract segmented object regions from masks.
9. Train TF-EfficientNetV2-M and ViT classifiers on extracted regions.
10. Evaluate with classification report, confusion matrix, and accuracy plots.

---

## Results / Output 📊
The project produces both visual and quantitative outputs:

- Segmentation overlays on real images
- Ground-truth vs prediction comparisons
- COCO metrics (bounding-box AP and segmentation AP/mAP)
- Class distribution and imbalance analysis
- Classification metrics from EfficientNetV2 and ViT:
  - Precision, recall, F1-score
  - Confusion matrices (raw and normalized)
  - Accuracy trends and training/validation curves

Note: exact metric values depend on training environment, random seed, and runtime configuration.

---

## How to Run the Project 🚀

### 1. Prerequisites
- Python 3.9+ recommended
- Jupyter Notebook or JupyterLab
- CUDA-enabled GPU recommended for training

### 2. Clone project
```bash
git clone <your-repo-url>
cd version2
```

### 3. Create environment and install basics
```bash
python -m venv .venv
.venv\Scripts\activate
pip install jupyter notebook
```

### 4. Open notebooks
```bash
jupyter notebook
```

### 5. Run notebooks
Run the notebook cells top-to-bottom.  
Each notebook installs its required packages in early cells.

Suggested order for full pipeline understanding:
1. Mask R-CNN workflow (segmentation and balanced split generation)
2. TF-EfficientNet workflow (region-based classification)
3. ViT workflow (region-based transformer classification)

### 6. Roboflow access
Set your Roboflow API key before download steps.  
For safer usage, prefer environment variables instead of hardcoding keys.

---

## Project Structure 🗂️
- mask-rcnn-model-v2.ipynb - End-to-end segmentation pipeline and evaluation
-tf-efficientnet-model-v2.ipynb - EfficientNetV2 classification pipeline
- vit-base-patch16-224-v2.ipynb - Vision Transformer classification pipeline

---

## Future Improvements 🔭
- Add automated experiment tracking (for reproducibility and comparison)
- Add model checkpoint/version management
- Add lightweight inference API (Flask/FastAPI) for deployment
- Add real-time webcam/mobile inference demo
- Extend to more waste classes and larger multi-location datasets
- Add semi-supervised or active-learning loop for faster dataset growth

---

## Conclusion
WasteSegCluster has demonstrated a practical and real world AI approach for better waste management by combining segmentation and classification in a single pipeline, which has helped turn raw images into clear and useful information about waste, making the project valuable for research and learning while also supporting future use in recycling and sustainability applications.
