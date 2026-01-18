from .curvepy import CurveletFrequencyGrid
from .filters import soft_threshold, compute_thresholds, calculate_psnr

import numpy as np
import skimage.color as color
from skimage.util import img_as_float

class CurveletDenoise:
    """
    A wrapper that handles Color Space conversion and channel looping
    """
    def __init__(self, fdct: CurveletFrequencyGrid):
        self.fdct = fdct

    def normalize_img(self, img):
        """
        Ensures image is float format (0.0 to 1.0).

        INPUTS:
            img: array, input image

        RETURNS:
            float_img: array, converted image
        """
        return img_as_float(img)

    def forward_yuv(self, rgb_image):
        """
        Converts RGB to YUV and runs Forward Transform on each channel.
        
        INPUTS:
            rgb_image: 3D array (H, W, 3)

        RETURNS:
            all_coeffs: list, contains coefficients for [Y, U, V] channels
        """

        # Convert rgb image into yuv
        yuv = color.rgb2yuv(rgb_image)

        # Split channels
        channels = [yuv[:, :, 0], yuv[:, :, 1], yuv[:, :, 2]]

        # Compute the transformation for each channel
        all_coeffs = []
        for channel in channels:
            coeffs = self.fdct.forward_transform(channel)
            all_coeffs.append(coeffs)
        
        return all_coeffs

    
    def inverse_yuv(self, all_coeffs):
        """
        Reconstructs YUV channels and converts back to RGB.
        
        INPUTS:
            all_coeffs: list, coefficients for [Y, U, V]

        RETURNS:
            rgb_image: 3D array (H, W, 3), final restored color image
        """
        reconstructed_channels = []

        # Compute inverse transformation for each channel
        for coeffs in all_coeffs:

            # Reconstruct single channel
            reconstructed_image = self.fdct.inverse_transform(coeffs)
            reconstructed_channels.append(reconstructed_image)
        
        # Stack back to (H, W, 3)
        yuv_image = np.stack(reconstructed_channels, axis=2)

        # Convert back to RGB
        rgb_image = color.yuv2rgb(yuv_image)

        # Ensure image is between 0.0 and 1.0
        return np.clip(rgb_image, 0, 1)
    
    def _denoise_channel(self, image_2d, sigma, multiplier):
        """Helper to process a single 2D channel"""
        coeffs = self.fdct.forward_transform(image_2d)
        thresholds = compute_thresholds(self.fdct, image_2d.shape, sigma, multiplier)
        
        new_coeffs = []
        for i, scale in enumerate(coeffs):
            new_scale = []
            for w, wedge in enumerate(scale):
                # Apply Soft Thresholding
                filtered = soft_threshold(wedge, thresholds[i][w])
                new_scale.append(filtered)
            new_coeffs.append(new_scale)
            
        return self.fdct.inverse_transform(new_coeffs)
        

    
    def denoise(self, image, sigma, multiplier=3.0):
        """
        Denoising of an image. Handles both greyscale and RGB images via Soft Thresholding and YUV transformation (for RGB images).
        
        INPUTS:
            rgb_image: 3D array, noisy input image
            sigma: float, estimated noise level (e.g. 0.1)
            multiplier: float, how aggressive to be (e.g. 1.5 or 3.0)

        RETURNS:
            clean_image: 3D array, denoised result
        """
        image = self.normalize_img(image)
        
        if image.ndim == 2:
            return self._denoise_channel(image, sigma, multiplier)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Breakdown image into coefficients
            all_coefficients = self.forward_yuv(image)

            # Denoise different channels differently  
            # Channel 0 = Y (Luma Light)
            # Channel 1, 2 are Cb, Cr (color)

            denoised_coeffs_all = []

            for i, channel_coeffs in enumerate(all_coefficients):
                threshold_list = compute_thresholds(self.fdct, image.shape[:2], sigma, multiplier)
                denoised_coeffs = []

                for j in range(len(channel_coeffs)):
                        
                    denoised_scale = []
                    

                    if j == 0:
                        denoised_scale = channel_coeffs[j]
                        denoised_coeffs.append(denoised_scale)
                        continue
                    
                    for w in range(len(channel_coeffs[j])):

                        data = channel_coeffs[j][w]
                        T = threshold_list[j][w]

                        clean_wedge = soft_threshold(data, T)
                        denoised_scale.append(clean_wedge)
                    
                    denoised_coeffs.append(denoised_scale)
                
                denoised_coeffs_all.append(denoised_coeffs)
            
            return self.inverse_yuv(denoised_coeffs_all)
        else:
            raise ValueError("Image must be 2D (Gray) or 3D (RGB)")
    
    def calculate_psnr_rgb(self, original, restored):
        return calculate_psnr(original, restored)
