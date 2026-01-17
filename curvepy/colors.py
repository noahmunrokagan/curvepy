from .curvepy import CurveletFrequencyGrid
from .filters import soft_threshold, compute_thresholds, calculate_psnr

import numpy as np
import skimage.color as color
from skimage.util import img_as_float

class ColorCuerveletDenoise:
    """
    A wrapper that handles Color Space conversion and channel looping
    """
    def __init__(self, fdct: CurveletFrequencyGrid):
        self.fdct = fdct

    def normalize_img(self, img):
        return img_as_float(img)

    def forward_yuv(self, rgb_image):

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
    
    def denoise(self, rgb_image, sigma, multiplier):
        """
        Denoising of a rgb image via soft threshold and yuv transformation
        
        :param self: Description
        :param rgb_image: Description
        :param sigma: Description
        """
        
        # Breakdown image into coefficients
        all_coefficients = self.forward_yuv(rgb_image)

        # Denoise different channels differently  
        # Channel 0 = Y (Luma Light)
        # Channel 1, 2 are Cb, Cr (color)

        denoised_coeffs_all = []

        for i, channel_coeffs in enumerate(all_coefficients):
            threshold_list = compute_thresholds(self.fdct, rgb_image.shape[:2], sigma, multiplier)
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
    
    def calculate_psnr_rgb(self, original, restored):
        return calculate_psnr(original, restored)
