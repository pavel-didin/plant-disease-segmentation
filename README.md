# Semantic Segmentation of Plant Leaf Diseases

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A hybrid approach for segmenting diseased areas on plant leaves, combining **K‑Nearest Neighbors (KNN)** for healthy tissue detection and **XGBoost** for classifying boundary regions. This method is designed for scenarios with limited annotated data and outperforms popular CNN‑based architectures under such constraints.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Predicting on a Single Image](#predicting-on-a-single-image)
  - [Evaluating on a Dataset](#evaluating-on-a-dataset)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Citation](#citation)
- [Author](#author)
- [License](#license)

## Overview
Accurate segmentation of plant diseases is crucial for automated agricultural monitoring. Deep learning models like U‑Net require large annotated datasets, while foundation models such as SAM need thousands of fine‑tuning examples. In contrast, our pipeline achieves reliable segmentation with only **~180 manually labeled regions** by leveraging classical computer vision and lightweight machine learning.

### How It Works
1. **Healthy Leaf Mask (KNN)**
   A KNN classifier trained on a small set of manually annotated rectangles separates healthy leaf pixels from everything else (background, shadows, diseased areas). This mask is refined using morphological operations and enclave removal.

2. **Internal Disease Detection**
   Pixels inside the leaf mask that were **not** classified as healthy are treated as diseased. These enclaves are cleaned with area‑based filtering.

3. **External / Boundary Disease Detection (XGBoost)**
   - Edge detection (Canny) and connected components are used to partition regions outside the healthy mask.
   - For each region, features are extracted: area, mean BGR/HSV values, per‑channel variance, and brightness.
   - A pre‑trained XGBoost classifier labels each region as diseased (class 0), healthy (1), shadow (2), or background (3).
   - Diseased regions are merged with the internal mask to produce the final segmentation.

4. **Post‑processing**
   Small isolated artifacts are removed using `merge_enclaves`, and contours are drawn on the original image.

The XGBoost model was trained iteratively: starting from a minimal set of labeled regions, erroneous predictions were added to the training set until performance stabilized.

## Project Structure
```text
plant-disease-segmentation/
├── README.md
├── LICENSE
├── requirements.txt
├── models/                          # Pre‑trained models
│   ├── knn_classifier.pkl
│   └── xgboost_model.pkl
├── data/
│   └── evaluation/                  # Evaluation dataset (optional)
│       ├── images/                  # 30 original leaf images
│       └── masks/                   # 30 ground‑truth binary masks
├── src/
│   ├── segmentation_utils.py        # Core algorithms
│   ├── predict.py                   # Single‑image prediction script
│   └── evaluate.py                  # Batch evaluation script
└── examples/                        # Example input/output images (optional)
```

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/pavel-didin/plant-disease-segmentation.git
   cd plant-disease-segmentation
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/macOS
   venv\Scripts\activate         # Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Predicting on a Single Image
```bash
python src/predict.py --image path/to/leaf.jpg --output result.jpg --mask_output mask.png
```
| Argument        | Description |
|-----------------|-------------|
| `--image`       | Path to input image (required) |
| `--output`      | Path to save output image with red contours |
| `--mask_output` | Path to save binary mask (0/255) |
| `--knn_model`   | Path to KNN model (default: `models/knn_classifier.pkl`) |
| `--xgb_model`   | Path to XGBoost model (default: `models/xgboost_model.pkl`) |

### Evaluating on a Dataset
```bash
python src/evaluate.py --images data/evaluation/images --masks data/evaluation/masks
```

| Argument      | Description |
|---------------|-------------|
| `--images`    | Directory containing original images |
| `--masks`     | Directory containing ground‑truth binary masks |
| `--visualize` | (Optional) Show each prediction pair (requires GUI) |

The script outputs average IoU, F1, and F2 scores.

### Results
Evaluated on a test set of 30 manually annotated grape leaf images affected by black rot, our pipeline achieved the following metrics:

| Metric | Value |
|--------|-------|
| IoU    | 0.60  |
| F1     | 0.74  |
| F2     | 0.75  |

**Comparison with deep learning approaches:**

**U‑Net**: Trained on the same amount of data, U‑Net struggled to distinguish diseased spots from shadows and background fragments, resulting in lower accuracy.

**SAM (Segment Anything Model)**: Requires fine‑tuning on thousands of annotated images to reach acceptable performance, which is impractical for this domain.

Segmentation accuracy is higher inside the leaf (internal enclaves found via KNN) than on the leaf boundaries. Inside the leaf, color is a reliable indicator of class membership. Near the edges, complex transitions, shadows, highlights, and background artifacts make the problem significantly harder.

### Limitations and Future Work
Performance on leaf boundaries remains the main bottleneck. Additional training examples of edge regions could improve the XGBoost classifier.

The current feature set could be extended with texture descriptors (e.g., LBP, GLCM) to better capture subtle differences between shadow and disease.

A lightweight neural network could be integrated for refining boundary predictions.

## Citation
```bibtex
@misc{plant-seg-2026,
  author       = {Didin Pavel},
  title        = {Semantic Segmentation of Plant Leaf Diseases Using KNN and XGBoost},
  year         = {2026},
  howpublished = {\url{https://github.com/pavel-didin/plant-disease-segmentation}}
}
```

## Author
Didin Pavel  
Email: didin.pa@phystech.edu  
GitHub: @pavel-didin

## License
This project is licensed under the MIT License – see the LICENSE file for details.
