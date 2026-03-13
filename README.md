# 🖼️ Digital Image Processing

A collection of three assignments implementing core image processing algorithms, developed for the course *Digital Image Processing* (2024 - 2025) at the Aristotle University of Thessaloniki, Department of Electrical and Computer Engineering.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Exercise 1 — Histogram Processing](#assignment-1--histogram-processing)
- [Exercise 2 — Edge & Circle Detection](#assignment-2--edge-detection--circle-finding)
- [Exercise 3 — Image Segmentation](#assignment-3--image-segmentation)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Structure

```
.
├── Exercise_1/                 # Histogram equalization & matching
│   ├── hist_utils.py
│   ├── hist_modify.py
│   ├── demo.py
│   └── report.pdf
│
├── Exercise_2/                 # Edge detection & Hough circles
│   ├── fir_conv.py
│   ├── sobel_edge.py
│   ├── log_edge.py
│   ├── circ_hough.py
│   ├── demo.py
│   └── report2.pdf
│
├── Exercise_3/                 # Graph-based image segmentation
│   ├── image_to_graph.py
│   ├── spectral_clustering.py
│   ├── n_cuts.py
│   ├── demo1.py
│   ├── demo2.py
│   ├── demo3a.py
│   ├── demo3b.py
│   ├── demo3c.py
│   └── report3.pdf
└

```

---

## Exercise 1 — Histogram Processing

Implements **histogram equalization** and **histogram matching** of grayscale images via three distinct strategies:

- **Greedy:** assigns input levels to output levels sequentially, always filling the current bin as soon as its target count is reached
- **Non-greedy:** delays the assignment when the remaining deficit is small, yielding a more balanced output histogram
- **Post-disturbance:** adds uniform noise ±d/2 to each pixel before the greedy pass, breaking the per-level tie problem and allowing finer-grained bin filling

`demo.py` applies all three modes to `input_img.jpg` for both equalization and histogram matching against `ref_img.png`, displaying input/output images and their histograms side-by-side.

---

## Exercise 2 — Edge Detection & Circle Finding

Implements a full **FIR convolution** engine and two edge detectors, then applies the **circular Hough transform** on top:

- **FIR convolution:** general 2D convolution with configurable origin tracking for both image and mask
- **Sobel detector:** computes the gradient magnitude using two 3×3 Sobel masks and thresholds it to a binary edge map
- **LoG detector:** applies the Laplacian-of-Gaussian kernel and finds edge locations at zero-crossings
- **Circular Hough transform:** votes in a 3D (a, b, ρ) accumulator space to detect circles in any binary edge image

`demo.py` runs on `basketball_large.png`, showing Sobel edge maps at multiple thresholds, a LoG edge map for comparison, and detected circles overlaid on the original image for five values of `V_min` using both Sobel+Hough and LoG+Hough pipelines.

---

## Exercise 3 — Image Segmentation

Implements **graph-based image segmentation** via Spectral Clustering and Normalized Cuts (N-Cuts), loading data from `dip_hw_3.mat`:

- **Image-to-graph:** converts any C-channel image into a fully-connected affinity matrix
- **Spectral clustering:** builds the unnormalised Laplacian, extracts the k smallest eigenvectors, and groups pixels with k-means
- **N-Cuts (non-recursive):** same pipeline but solves the generalised eigenproblem instead, giving a normalised cut criterion
- **N-Cuts (recursive):** repeatedly bisects the graph (k=2) and stops a branch when cluster size < T1 or the Ncut metric > T2

### Demos

| Script | Description |
|--------|-------------|
| `demo1.py` | Spectral clustering on pre-built affinity matrix `d1a` for k = 2, 3, 4 |
| `demo2.py` | Spectral clustering on RGB images `d2a`, `d2b` for k = 2, 3, 4 |
| `demo3a.py` | Non-recursive N-Cuts on `d2a`, `d2b` for k = 2, 3, 4; comparison with spectral clustering |
| `demo3b.py` | One-step recursive N-Cuts (k=2) on both images; Ncut metric reported |
| `demo3c.py` | Full recursive N-Cuts with T1=5, T2=0.20; comparison with spectral clustering and non-recursive N-Cuts |

---

## Installation

### Requirements

- Python 3.9+
- NumPy
- SciPy
- scikit-learn
- matplotlib
- Pillow

```bash
pip install numpy scipy scikit-learn matplotlib pillow
```

### Required Data Files

```
Exercise_1/  →  input_img.jpg, ref_img.png
Exercise_2/  →  basketball_large.png
Exercise_3/  →  dip_hw_3.mat
```

---

## Usage

Run each demo script directly from its assignment folder:

```bash
# Assignment 1
cd Exercise_1/ && python demo.py

# Assignment 2
cd Exercise_2/ && python demo.py

# Assignment 3 — run demos in order
cd Exercise_3/
python demo1.py
python demo2.py
python demo3a.py
python demo3b.py
python demo3c.py
```
