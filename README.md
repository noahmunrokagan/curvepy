# CurvePy: Fast Discrete Curvelet Transform (FDCT)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

**CurvePy** is a pure Python "clean room" implementation of the **Fast Discrete Curvelet Transform** via Uniform Wrapping outlined in Emmanuel Candès 2005 paper [Fast Discrete Curvelet Transforms](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://math.mit.edu/icg/papers/FDCT.pdf)

Unlike wavelets, which represent images using points, Curvelets represent images using oriented edges. This makes CurvePy (or the FDCT) a powerful tool for sparse representations of smooth curves, and highly effective at denoising while preserving sharp boundaries.

---

## Results

### Denoising

### Grayscale images (2-D)

### RGB images (3-D)
Curvepy uses a color denoising engine that operatues in the YUV color space (chosen to preserve structural details) while filtering chromatic noise.

## Features
* **Fast Discrete Curvelet Transform (FDCT):** Implemented via the "wrapping" method outlined in Candès et al. 2005 for computational efficiency ($O(N^2 \log N)$).
* **Color Support:** wrapper specifically for RGB images using **YUV** separation. Processing Luma and Chroma channels independently.
* **Thresholding Design:** 
    * **Soft Thresholding:** For artifact-free restoration.
    * **Monte Carlo Calibration:** Estimates noise levels per wedge and per scale.
* **Module Design:** Separates geometry computations, windowing and filtering logic into different modules.

---

## Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/yourusername/curvepy.git](https://github.com/yourusername/curvepy.git)
cd curvepy
pip install -r requirements.txt
```

## Usage
1. Basic Transformation (Grayscale image)
```python
import matpolotlib.pyplot as plt
import skimage.data as data
from curvepy.curvepy import CurveletFrequencyGrid


# Initialize the Transformation engine (512x512 grid, 4 scales)
fdct = CurveletFrequencyGrid(N=512, scales=4)

# Load Image
image = data.camera()

# 1. Forward Transform
coefficients = fdct.forward_transform(image)

# 2. Inverse Transform
reconstructed_image = fdct.inverse_transform(coefficients)

# 3. Plot results
plt.figure(figsize=(12,4))
plt.suptitle("Original vs Reconstructed Image")

plt.subplot(1,2,1)
plt.imshow(image, plt.cmap.gray)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(reconstructed_image, plt.cmap.gray)
plt.title("Restored Image")

plt.tight_layout()
plt.show()
```

2. Basic Transformation and Denoising (Grayscale image)
```python
import matpolotlib.pyplot as plt
import skimage.data as data
from curvepy.curvepy import CurveletFrequencyGrid
from curvepy.denoise import CurveletDenoise


# Initialize the Transformation engine (512x512 grid, 4 scales)
fdct = CurveletFrequencyGrid(N=512, scales=4)
denoise_engine = CurveletDenoise(fdct)

# Load Image
image = data.camera()

# 1. Forward Transform
coefficients = fdct.forward_transform(image)

# 2. Denoise
restored_img = denoise_engine.denoise(noisy_img, sigma, multiplier)
psnr = denoise_engine.calculate_psnr_rgb(img, restored_img)

# 3. Plot results
plt.figure(figsize=(12,4))
plt.suptitle(f"Original vs restored img for soft thresholding, PSNR = {psnr:.2f} dB, multiplier = {multiplier}")

plt.subplot(1, 2, 1)
plt.imshow(img, cmap=plt.cm.gray)
plt.title("original img")

plt.subplot(1, 2, 2)
plt.imshow(restored_img, cmap=plt.cm.gray)
plt.title("restored img")

plt.tight_layout()
plt.show()
```

3. Denoising a Color Image
```python
import matplotlib.pyplot as plt
import skimage.data as data
from curvepy.curvepy import CurveletFrequencyGrid
from curvepy.denoise import ColorCuerveletDenoise

# Setup
fdct = CurveletFrequencyGrid(N=512, scales=4)
denoise_engine = CuerveletDenoise(fdct)
image = denoise_engine.normalize_img(data.astronaut())

# Apply denoising
restored_image = denoise_engine.denoise(image, sigma=0.1, multiplier=1.5)
psnr = denoise_engine.calculate_psnr_rgb(image, restored_image) # Calculate peak SNR value between two images

# Plot results
plt.figure(figsize=(12,4))
plt.suptitle(f"Original vs Restored image via Soft Thresholding, PSNR = {psnr:.2f} dB, multiplier = {multiplier}")

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("original img")

plt.subplot(1, 2, 2)
plt.imshow(restored_img)
plt.title("restored img")

plt.tight_layout()
plt.show()
```

## Project Structure
curvepy/\
├── curvepy.py       # Core Engine: FDCT & IFDCT implementation \
├── windows.py       # Math: Meyer Window functions (Phi, Psi, V) \
├── filters.py       # Tools: Thresholding logic & Monte Carlo calibration \
└── denoise.py        # App: YUV Color wrapper & Denoising pipeline

## Theory + How it Works:
Standard 2D Wavelets are isotropic, ie. they treat all directions equally. This doens't work well for images where the curves/outlines of the shapes are what are important to be preserved. This creates a blocky or ringing effect when trying to represent a smooth curve.

**Curvelets** are anisotropic "needles" that exist at different scales and angles
* **Scales:** Captures details of different sizes.
* **Angles:** Capture the direction of the geometry.

This allows CurvePy to represent a curved edge with very few coefficients relative to the aformentioneed 2D wavelets, making it ideal for compression and restoration tasks while preserving the geometry is important :)

To understand why this matters, we visualized the inner workings of the transform below:

Image
![Astronaut Image](assets/astronaut.png)
Transformation Process for Above Image
![Curvelet Theory Grid](assets/curvelet_theory_image.png) 


### The 4-Step Pipeline (Explained)

1.  **The Map (Top-Left):**
    The Frequency Plane is tiled into "Wedges." Each wedge represents a specific combination of **Scale** (how thick the feature is) and **Angle** (which way it points).
    * *Center (Grey):* Low-frequency background (blurry shapes).
    * *Outer Rings (Colors):* High-frequency details (sharp edges).

2.  **The Filter (Top-Right):**
    To detect specific features, we activate just **one single wedge**. In this example, we selected **Scale 3, Wedge 0**. This filter is specialized to find "Medium-Fine details that are Horizontal."

3.  **The Needle (Bottom-Left):**
    This is what that single filter looks like in the real world (Spatial Domain). It isn't a point—it's a **Needle**.

4.  **The Response (Bottom-Right):**
    When we drag this Needle across a real photo (The Astronaut), it lights up (turns black) *only* where it finds matching geometry.
    * Notice the top of the helmet and the flag stripes are clearly visible because they are horizontal.
    * The vertical rocket boosters in the background are invisible. The horizontal needle doesn't notice the vertical lines.

**Summary:** By adding up thousands of these "Needles" at every possible angle and size, CurvePy reconstructs the perfect image—minus the noise.

## Licence
MIT Licence. Free to use for academic and personal projects.