# Image ReTransformation via Block-Based Sinkhorn Optimal Transport

This project implements pixel-level image transformation using entropy-regularized optimal transport (Sinkhorn iterations). Given a source and target image, the system computes a transport plan in a joint color–spatial feature space, extracts a permutation mapping, and generates both a transformed image and a smooth morphing video.

The implementation includes multi-scale refinement, flow-field smoothing, and optional GPU acceleration.

---

## Overview

We treat images as discrete probability distributions over pixels. Each pixel is represented in a combined feature space of color and spatial coordinates. The optimal transport plan between source and target distributions is computed using entropy-regularized Sinkhorn iterations.

To improve scalability and visual smoothness, the pipeline incorporates:

- Multi-scale coarse-to-fine optimal transport
- Block-based refinement
- Deterministic permutation extraction
- Displacement field smoothing
- Continuous-time fluid morph rendering

---

## Method

### Pixel Feature Representation

Each pixel is embedded into a 5D feature space:

[ R, G, B, λx, λy ]

- RGB values are normalized to [0, 1]
- (x, y) are normalized spatial coordinates
- λ controls the strength of spatial regularization

This encourages transport plans that remain spatially coherent.

---

### Entropy-Regularized Optimal Transport

We compute the Sinkhorn transport plan:

P = diag(u) K diag(v)

Where:

K = exp(-C / ε)

- C is the pairwise cost matrix (L2 distance in feature space)
- ε is the entropy regularization strength
- u and v are iteratively updated scaling vectors

Sinkhorn iterations enforce row and column marginal constraints.

---

### Transport Plan to Permutation

The soft transport matrix is converted into a one-to-one mapping via greedy selection of maximum transport mass entries. This produces a deterministic pixel permutation.

---

### Multi-Scale Refinement

To scale beyond naive full-resolution optimal transport:

1. Downsample images to a coarse resolution
2. Compute Sinkhorn optimal transport
3. Propagate barycentric mappings to higher resolution
4. Refine correspondences to reduce block artifacts

This significantly improves computational feasibility and output quality.

---

### Fluid Morph Generation

A displacement field is constructed from the extracted permutation. The field is:

- Smoothed using iterative relaxation
- Interpolated over time
- Used to warp the source image continuously toward the target

The system outputs a morphing video illustrating the transformation.

---

## Pipeline Usage

```python
transform_image_to_target(
    input_path="source.png",
    target_path="target.png",
    out_path="output.png",
    video_path="morph.mp4",
    work_size=(128,128),
    out_size=(128,128),
    pos_weight=0.3,
    use_gpu=True
)
```

## Parameters

epsilon: Entropy regularization strength
n_iters: Number of Sinkhorn iterations
pos_weight: Spatial coordinate weight
work_size: Resolution used for OT computation
out_size: Final output resolution
smooth_iters: Flow-field smoothing iterations
use_gpu: Enable GPU acceleration (PyTorch)

## Requirements

-Python 3.9+
-PyTorch
-NumPy
-Pillow
-imageio

## Install dependencies:

pip install torch numpy pillow imageio

## Core Components

sinkhorn_transport() – entropy-regularized optimal transport
build_pixel_features() – RGB + spatial embedding
compute_cost_matrix() – feature-space cost computation
multi_scale_ot_permutation() – coarse-to-fine mapping
smooth_flow_field() – displacement field regularization
create_fluid_morph_video() – continuous morph rendering

## Output

The system produces:

A transformed image generated via pixel permutation

A smooth morphing video visualizing the transport process

## Future Improvements

Log-domain stabilized Sinkhorn iterations

Exact bipartite matching (Hungarian algorithm)

Sliced Wasserstein approximations for scalability

Extension to video-to-video transport

Learned feature embeddings instead of raw RGB
