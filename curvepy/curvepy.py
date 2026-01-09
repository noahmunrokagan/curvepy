import numpy as np
import matplotlib.pyplot as plt
import curvepy.windows as windows

# Standard Curvelet setups usually use 8 wedges per quadrant at the 2nd coarse scale
DEFAULT_WEDGES = 4

class CurveletFrequencyGrid():
    def __init__(self, N: int, scales: int):
        self.N = N
        self.scales = scales
        
        # Coordinate Grid (Use float to avoid integer division issues)
        # We use a slight offset or 'eps' to avoid division by zero errors in Slopes
        self.Y, self.X = np.mgrid[-N//2:N//2, -N//2:N//2].astype(float)
        
        # Add epsilon to avoid divide-by-zero (inf is okay, but nan is annoying)
        self.X[self.X == 0] = 1e-10 
        self.Y[self.Y == 0] = 1e-10

        # Pre-compute Radius and Slopes
        # R = max(|x|, |y|) is the "L-infinity" norm used for square shells
        self.R = np.maximum(np.abs(self.X), np.abs(self.Y))
        self.Slopes_EW = self.Y / self.X
        self.Slopes_NS = self.X / self.Y

        # 2. Quadrant Masks
        self.Quadrants = {
            "East":  (self.X > 0) & (np.abs(self.Y) <= self.X),
            "West":  (self.X < 0) & (np.abs(self.Y) <= np.abs(self.X)),
            "North": (self.Y < 0) & (np.abs(self.X) <= np.abs(self.Y)),
            "South": (self.Y > 0) & (np.abs(self.X) <= self.Y)
        }

    def _get_scale_bounds(self, scale_idx: int):
        """Returns the integer radius boundaries (inner, outer) for a scale."""
        center_idx = self.N // 2
        
        # Inverse logic: Scale 0 is coarsest, Scale (scales-1) is finest
        inverse_scale = (self.scales - 1) - scale_idx
        
        # Outer boundary of this scale
        radius_outer = self.N // (2 ** (inverse_scale + 1))
        
        # Inner boundary (which is the outer boundary of the previous scale)
        # If scale_idx is 0, inner is 0.
        if scale_idx == 0:
            radius_inner = 0
        else:
            radius_inner = self.N // (2 ** (inverse_scale + 2))
            
        return max(1, int(radius_inner)), max(1, int(radius_outer))
    
    def get_radial_window(self, scale_idx):
        """
        Generate the Radial 'Donut' Mask.
        Uses Difference of Squares: sqrt(Phi_outer^2 - Phi_inner^2)
        """
        r_inner, r_outer = self._get_scale_bounds(scale_idx)

        # Outer Low-Pass (Phi) We normalize R so that r_outer maps to 1.0 (where Phi drops to 0)
        # Note: meyer_phi drops from 1->0 between 0.5 and 1.0.
        phi_outer = windows.meyer_phi(self.R / r_outer)
        
        # Inner Low-Pass (Phi)
        if scale_idx == 0:
            # Coarsest scale is just the low-pass itself
            return phi_outer
        else:
            # We want the window to be 0 inside r_inner.
            # Phi(R/r_inner) is 1 inside r_inner.
            phi_inner = windows.meyer_phi(self.R / r_inner)
            
            # The "Shell" is the region between them.
            # We use Sqrt(Outer^2 - Inner^2) to preserve energy.
            # Clip to 0 to avoid negative sqrts due to float precision
            return np.sqrt(np.maximum(0, phi_outer**2 - phi_inner**2))
        
    def get_angular_window(self, quadrant_name, slope_min, slope_max):
        """
        Generate the Angular 'Wedge' Mask.
        Applies meyer_v centered on the wedge.
        """
        # Select correct slope grid
        if quadrant_name in ["East", "West"]:
            slopes = self.Slopes_EW
        else:
            slopes = self.Slopes_NS

        # Define Wedge Geometry
        slope_center = (slope_min + slope_max) / 2.0
        slope_width = slope_max - slope_min

        # Normalize Slopes to [-1, 1] domain for the window
        normalized_slope = (slopes - slope_center) / slope_width
        
        # Apply Window
        # This will create a "beam" extending from the origin
        return windows.meyer_v(normalized_slope)
    
    def get_wedge_filter(self, scale_idx, wedge_idx_in_scale):
        """
        Returns the Soft Wedge Filter for a specific scale and wedge index.
        Handles mapping global index -> Quadrant -> Slope.
        """
        # Get Radial Donut
        radial_mask = self.get_radial_window(scale_idx)
        
        # If scale 0 (Center), just return the radial mask (low pass)
        if scale_idx == 0:
            return radial_mask

        # Determine Wedge Slope Bounds
        boundaries = self._get_wedge_slope_ranges(scale_idx)
        # boundaries has N+1 items for N wedges.
        wedges_per_quadrant = len(boundaries) - 1
        
        # Map Global Index to Quadrant & Slope
        # We assume standard counter-clockwise order: East -> North -> West -> South
        quad_names = ["East", "North", "West", "South"]
        
        quad_idx = wedge_idx_in_scale // wedges_per_quadrant
        slope_idx = wedge_idx_in_scale % wedges_per_quadrant
        
        # Safety check for index out of bounds
        if quad_idx >= 4:
            raise ValueError(f"Wedge Index {wedge_idx_in_scale} exceeds max for Scale {scale_idx}")

        quadrant = quad_names[quad_idx]
        
        # Get Slope Range
        # Note: Slope definition might need reversal for certain quadrants to maintain
        # continuous rotation, but for visualization, direct mapping is fine.
        s_min = boundaries[slope_idx]
        s_max = boundaries[slope_idx+1]
        
        # Get Angular Beam
        angular_mask = self.get_angular_window(quadrant, s_min, s_max)
        
        # Get Quadrant Hard Mask (To cut off the wrap-around slopes)
        quadrant_mask = self.Quadrants[quadrant]
        
        # Combine
        return radial_mask * angular_mask * quadrant_mask
    
    def _get_wedge_slope_ranges(self, scale_idx: int):
        if scale_idx == 0: return None
        steps = int((scale_idx - 1) // 2) 
        num_wedges = DEFAULT_WEDGES * (2 ** steps)
        return np.linspace(-1.0, 1.0, int(num_wedges) + 1)
    
    def get_wedge_dimensions(self, scale_idx):
        """
        Returns optimal (L1, L2) rectangle size for a wedge at this scale.
        L1 is 'Lengeth' (radial), L2 is 'Width' (Angular)
        """
        if scale_idx == 0:
            # Coarse scale is just a square in the center
            # Pad slightly for safety
            radius, _ = self._get_scale_bounds(0)
            dimension = (radius * 2) + 1
            return dimension, dimension
        
        # We use parabolic scaling for finer scales
        inverse_scale_idx = (self.scales - 1) - scale_idx

        # Dimensions derived from Candès et al. 2005
        L1 = 4 * self.N // (2 ** (inverse_scale_idx + 2))
        L2 = self.N // (2 ** (inverse_scale_idx//2 + 1)) # Parabolic scaling

        return int(L1), int(L2)
    
    def _get_wedge_center(self, scale_idx, wedge_idx):
        """Calculates the geometric center of the wedge using the MASK."""
        mask = self.get_wedge_filter(scale_idx, wedge_idx)
        arg_max = np.unravel_index(np.argmax(mask), mask.shape)
        return arg_max # (cy, cx)
    
    def wrap_wedge(self, wedge_data, scale_idx, wedge_idx):
        """
        Cuts out the "glowing trapezoid" and wraps it into rectangle L1 x L2
        
        :param self: Description
        :param wedge_data: Description
        :param scale_idx: Description
        :param wedge_idx: Description
        """
        L1, L2 = self.get_wedge_dimensions(scale_idx)

        # Find the approximate center of the wedge (to be changed later)
        cy, cx = self._get_wedge_center(scale_idx, wedge_idx)

        # Cut out rectangle, handle indices moved by np.roll
        shift_x_center = (self.N // 2) - cx
        shift_y_center = (self.N // 2) - cy

        centered_data = np.roll(wedge_data, shift_y_center, axis=0)
        centered_data = np.roll(centered_data, shift_x_center, axis=1)

        # Slice middle pixels
        start_x = (self.N // 2) - (L2 // 2)
        start_y = (self.N // 2) - (L1 // 2)

        small_wedge = centered_data[start_y:start_y + L1, start_x:start_x + L2]

        return small_wedge
    
    def unwrap_wedge(self, wrapped_data, scale_idx, wedge_idx):
        """
        Reverses the wrapping
        Puts the small L1xL2 wedge back into the N x N grid.
        """
        L1, L2 = wrapped_data.shape

        # Create the target grid
        big_grid = np.zeros((self.N, self.N), dtype=complex)

        # Place small wedge in center of grid
        start_y = (self.N // 2) - (L1 // 2)
        start_x = (self.N // 2) - (L2 // 2)

        big_grid[start_y:start_y + L1, start_x:start_x + L2] = wrapped_data

        # Determine shift
        cy, cx = self._get_wedge_center(scale_idx, wedge_idx)

        shift_y_center = (self.N // 2) - cy
        shift_x_center = (self.N // 2) - cx

        # Unroll
        unwrapped_grid = np.roll(big_grid, -shift_y_center, axis=0)
        unwrapped_grid = np.roll(unwrapped_grid, -shift_x_center, axis=1)

        return unwrapped_grid
    

    def forward_transform(self, image):
        """
        Perform fast discrete curvelet transform via wrapping
        """
        # Compute 2-D array fast fourier transform, and shift it (since our grid has (0,0) at the centre)
        image_frequency = np.fft.fftshift(np.fft.fft2(image))

        coefficients = []
        for scale_idx in range(self.scales):
            scale_coefficients = []

            # Low-pass
            if scale_idx == 0:
                mask = self.get_wedge_filter(0, 0)
                data = image_frequency * mask

                dimensions = self.get_wedge_dimensions(0) # Returns L x L for coarse rectangle

                # Cut out center (image already centered so crop is simple)
                center_y, center_x = self.N // 2, self.N // 2
                radius = dimensions[0] // 2

                # Get slice indices
                s_row = slice(center_y - radius, center_y + radius + 1)
                s_col = slice(center_x - radius, center_x + radius + 1)

                wrapped_data = data[s_row, s_col]

                # Inverse fast fourier transformation back to spatial dimensions
                # Note: we shift back since we shifted to start

                coeffs = np.fft.ifft2(np.fft.ifftshift(wrapped_data))
                scale_coefficients.append(coeffs)
                coefficients.append(scale_coefficients)
                continue

            # Handle wedges (scales 1 -> N-1)
            boundaries = self._get_wedge_slope_ranges(scale_idx)

            # Total wedges = 4 quadrants * wedges per quadrant
            num_wedges = (len(boundaries) - 1) * 4

            for wedge_idx in range(num_wedges):
                # Generate mask
                mask = self.get_wedge_filter(scale_idx, wedge_idx)

                # Apply mask
                wedge_data = image_frequency * mask

                # Wrap data
                wrapped_data = self.wrap_wedge(wedge_data, scale_idx, wedge_idx) # Tiny rectangle centered at (0, 0)

                # Inverse transformation
                coeffs = np.fft.ifft2(np.fft.ifftshift(wrapped_data))

                scale_coefficients.append(coeffs)

            coefficients.append(scale_coefficients)

        return coefficients
    
    def inverse_transform(self, coefficients):
        """
        Performs inverse fast discrete curvelet transform via wrapping
        
        INPUTS:
        coefficients: List of lists 'coefficients' from forward_transform
        
        OUTPUT:
        (N, N) reconstructed image
        """

        reconstructed_frequency = np.zeros((self.N, self.N), dtype=complex)

        for j, scale_coeffs in enumerate(coefficients):

            if j == 0:
                # FFT to get back to frequency
                data = scale_coeffs[0]
                frequency_data = np.fft.fftshift(np.fft.fft2(data))

                # Uncrop to get back to center
                L = frequency_data.shape[0]
                temporary_grid = np.zeros((self.N, self.N), dtype=complex)

                center = self.N // 2
                radius = L // 2

                s_row = slice(center - radius, center + radius + 1)
                s_col = slice(center - radius, center + radius + 1)

                # Handle odd/even shape mismatch
                temporary_grid[s_row, s_col] = frequency_data

                # Apply window
                mask = self.get_wedge_filter(0, 0)
                reconstructed_frequency += temporary_grid * mask
                continue

            # Handle wedges
            num_wedges = len(scale_coeffs)

            for wedge_idx in range(num_wedges):
                # FFT coefficients to get to frequency
                spatial_data = scale_coeffs[wedge_idx]

                # Shift to center
                wrapped_frequency = np.fft.fftshift(np.fft.fft2(spatial_data))

                # Unwrap
                unwrapped_frequency = self.unwrap_wedge(wrapped_frequency, j, wedge_idx)

                # Apply window
                mask = self.get_wedge_filter(j, wedge_idx)
                reconstructed_frequency += unwrapped_frequency * mask
            
        # Final inverse transform
        reconstructed_image = np.fft.ifft2(np.fft.ifftshift(reconstructed_frequency))

        return np.real(reconstructed_image)




# --- VISUALIZATION ---
if __name__ == "__main__":
    # Use scales=6 for 512x512 to match the paper's 'tight' center
    # Use 4 wedges per quadrant (16 total) as the base to match the paper's coarse scale
    fdct = CurveletFrequencyGrid(N=512, scales=6) 
    
    print("Building Grid...")
    all_wedges = fdct.build_grid()
    
    # VISUALIZATION FIX:
    # Use random colors so neighbors don't blend together
    viz_map = np.zeros((512, 512))
    
    # Shuffle indices to ensure random colors
    # We add 10 to start above 0 (background)
    import random
    indices = list(range(len(all_wedges)))
    random.shuffle(indices)

    for i, mask in enumerate(all_wedges):
        # Assign a random discrete value
        viz_map[mask] = indices[i] + 10 
        
    plt.figure(figsize=(10, 10))
    plt.title("Curvelet Frequency Tiling (Corrected Viz)")
    # 'nipy_spectral' is a high-contrast rainbow map
    plt.imshow(viz_map, cmap='nipy_spectral', origin='upper') 
    plt.axis('off')
    plt.show()