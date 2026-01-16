from .curvepy import CurveletFrequencyGrid
from .filters import soft_threshold, compute_thresholds

class ColorCuerveletDenoise:
    """
    A wrapper that handles Color Space conversion and channel looping
    """
    def __init__(self, fdct: CurveletFrequencyGrid):
        self.fdct = fdct

    
    def forward_ycbcr(self, rgb_image):
        return None
    
    def inverse_ycbcr(self, color_coeffs):
        return None
    
    def denoise(self, rgb_image, sigma, threshold_function):
        return None