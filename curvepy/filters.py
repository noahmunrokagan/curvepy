import numpy as np
from .curvepy import CurveletFrequencyGrid


def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def hard_threshold(data, threshold):

    mask = np.abs(data) > threshold

    return data * mask

def soft_threshold(data, threshold):
    magnitude = np.abs(data) 
    new_magnitude = np.maximum(0, magnitude - threshold)
    return np.sign(data) * new_magnitude

def calculate_psnr(orignal, restored):
    """
    Calculates peak signal to noise ratio measured in decibels (dB)
    
    INPUTS:
        original: normalized image in its original form (pre transformation/filtering)
        restored: normalized restored image (after transfomation, filtering, inverse transform)
    
    OUTPUTS:
        psnr: float, value indicating the peak SNR value in dB
    """
    mse = np.mean((orignal - restored) ** 2)

    if mse == 0:
        return 100
    
    # asssuming image is 0.0 -> 1.0
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr

def compute_thresholds(freq_grid: CurveletFrequencyGrid, image_shape, sigma_pixel, k=3.0):
    """
    Monte Carlo Noise Estimator for normlaized rgb images
    
    INPUTS:
        freq_grid: CurveletFrequencyGrid object
        image_shape: shape of the image (H, W)
        sigma_pixel: Noise level added to pixels
        k: threshold multiplier. Lower = more detail, higher = less noise

    OUTPUTS:
        thresholds: List of lists containing threshold for every wedge
    """

    dummy_noise = np.random.normal(0, sigma_pixel, size=image_shape)
    noise_coeffs = freq_grid.forward_transform(dummy_noise)
    thresholds = []

    for scale in noise_coeffs:
        scale_thresholds = []
        for wedge in scale:

            sigma_wedge = np.std(wedge)

            T = k * sigma_wedge # In Gaussian statistics, 99.7% of noise is within 3 sigmas.
            scale_thresholds.append(T)

        thresholds.append(scale_thresholds)
    
    return thresholds

