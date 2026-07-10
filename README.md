# DarkIR: Low-Light Image Restoration & Enhancement Pipeline

## IIT BHU Internship Project | CVPR 2025 Paper Implementation


## Overview

This project presents a two-phase implementation
of DarkIR (CVPR 2025) — a state-of-the-art
low-light image restoration model.

* Phase 1:Reproduced the DarkIR source code
and validated results on the standard LOLv2-Synthetic
benchmark dataset.

*Phase 2:Extended DarkIR with a custom
post-processing pipeline (**CLAHE + Unsharp Masking**)
and applied it to low-light degraded images
derived from the Ancient Chinese Classics dataset.

Note: The dataset contains mixed images
(not purely manuscripts) and includes some
watermarks present in the original data.

## Paper Reference

> **DarkIR: Robust Low-Light Image Restoration**  
> Daniel Feijoo, Juan C. Benito, Alvaro Garcia, Marcos V. Conde  
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025*  
> GitHub: https://github.com/cidautai/DarkIR
> ```python
dark_img = np.power(img/255.0, 2.5) * 255

## Project Structure

DarkIR-LowLight-Image-Restoration/
│
├── README.md
│
├── notebooks/

│   ├── 1_DarkIR_SourceCode_Run.ipynb

│   └── 2_DarkIR_Implementation.ipynb
│
├── results/

│   ├── source_code/

│   │   └── sample_comparison.png

│   └── implementation/

│       ├── sample_1.png

│       ├── sample_2.png

│       ├── sample_3.png
      


## Phase 1: Source Code Reproduction

### What I Did

Step 1 — Environment Setup
- Google Colab (T4 GPU)
- Cloned official DarkIR repository
- Installed dependencies: basicsr, lpips, ptflops, pytorch-msssim, einops, PyYAML


Step 2 — Dataset (LOLv2-Synthetic)

- Used official LOLv2-Synthetic dataset  
- Test set: 100 paired images  
  - `Low/`: dark images  
  - `Normal/`: ground truth  



Step 3 — Pretrained Weights

- Used official pretrained weights:
  - `DarkIR_allLOL.pt` → evaluation
  - `DarkIR_384.pt` → inference


**Step 4 — Model Setup

- DarkIR-m configuration:
  - Width: 32
  - Parameters: 3.31M
  - Encoder: [1,2,3]
  - Decoder: [3,1,1]
  - Dilations: [1,4,9]


Step 5 — Qualitative Evaluation

- Used a few external dark images (due to dataset size constraints)
- Generated visual comparisons


Step 6 — Quantitative Evaluation

- Evaluated on 100 test images  
- Metrics computed:

| Metric    | Our Result | Paper Result | Difference |
|----------|-----------|-------------|------------|
| PSNR     | 25.22 dB  | 25.54 dB    | -0.32      |
| SSIM     | 0.938     | 0.934       | +0.004     |

Confirms correct reproduction of the paper


## Phase 2: Custom Implementation

### Motivation

Low-light degradation and document degradation share similarities:
- Low illumination  
- Noise  
- Blur  

Thus, DarkIR can be extended beyond natural images
to document-style images.

---

## Dataset

**Ancient Chinese Classics Image Dataset (Kaggle)**

- Total Images: 20  
- Mixed content (not purely manuscripts)  
- Some images contain **watermarks (original dataset issue)**  


## Synthetic Low-Light Generation

Since dataset was not originally low-light:

- Applied **Gamma Transformation (γ ≈ 2.5–2.8)**
- Reduced brightness
- Added slight noise



# Proposed Pipeline
Input Image (Dark) → DarkIR → CLAHE → Sharpening → Final Output

Step 1: DarkIR
Low-light enhancement
Denoising
Structure restoration

Step 2: CLAHE (My Contribution)
Enhances local contrast
Applied in LAB color space
Prevents over-enhancement
Improves visibility of details/text

Step 3: Unsharp Masking (My Contribution)
Enhances edges
Improves clarity
Formula:
output = 1.5 × image − 0.5 × blurred


# Implementation Steps

1-Load pretrained DarkIR model

2-Load dataset from Google Drive

3-Apply artificial darkening

4-Run DarkIR inference

5-Apply CLAHE

6-Apply sharpening

7-Save outputs

8-Generate visual comparisons

9-Perform brightness analysis

10-Results

11-Stage-wise Output

12-Stage	Description

13-Input	Dark image

14-DarkIR	Bright + denoised

15-CLAHE	Contrast enhanced

16-Final	Sharp + clear output

#### Quantitative Analysis (20 Images)
  Image                  Dark Input   DarkIR   Pipeline    Gain
----------------------------------------------------------
  Img (1).jpg                  55.4    129.0      138.8   +83.4
  Img (10).jpg                  3.7     27.1       46.2   +42.5
  Img (11).jpg                 44.0     98.6      121.6   +77.5
  Img (12).jpg                 30.1     72.5       91.4   +61.4
  Img (13).jpg                 41.8    117.9      131.3   +89.5
  Img (14).jpg                 38.7     87.7      108.4   +69.7
  Img (15).jpg                 13.2     51.5       80.4   +67.2
  Img (16).jpg                 17.2     56.3       76.2   +59.0
  Img (17).jpg                 21.1     69.7      101.1   +79.9
  Img (18).jpg                 37.1     88.8      114.4   +77.3
  Img (19).jpg                 44.5     98.8      120.9   +76.4
  Img (2).jpg                  16.3     59.6       75.4   +59.1
  Img (20).jpg                 20.7     68.7      104.9   +84.2
  Img (3).jpg                  17.7     52.4       80.1   +62.4
  Img (4).jpg                  33.0     86.3      113.2   +80.2
  Img (5).jpg                  34.1     91.2      113.2   +79.0
  Img (6).jpg                  74.1    154.3      161.8   +87.7
  Img (7).jpg                  35.5     86.9      112.4   +76.9
  Img (8).jpg                  24.4     58.2       76.1   +51.7
  Img (9).jpg                  15.0     49.7       69.4   +54.5
==========================================================
  AVERAGE                      30.9     80.2      101.8   +71.0

#### Tools
Python
PyTorch
OpenCV
NumPy
Matplotlib
basicsr

#### Key Highlights
Successfully reproduced DarkIR results
Designed a multi-stage restoration pipeline
Combined deep learning + image processing
Performed both qualitative & quantitative analysis

#### Limitations
Dataset contains mixed images
Some images include watermarks
Synthetic darkening ≠ real low-light conditions
Connection to IIT BHU Internship
Applied DarkIR in image restoration domain
Built pipeline relevant to document/image enhancement
Demonstrates real-world applicability of research models

#### Author
Anjali Singh Yadav
B.Tech AI & ML
IIMT College of Engineering
IIT BHU Internship — Image Restoration

#### Acknowledgement

Based on DarkIR (CVPR 2025) by Feijoo et al.
Official implementation and pretrained weights used.












