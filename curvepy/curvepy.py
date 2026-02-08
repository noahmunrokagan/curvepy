import numpy as np
import curvepy.windows as windows
try:
    from . import inner_loop
    CYTHON_AVILABLE = True
    print("Curvepy accelerated with Cython")
except ImportError:
    CYTHON_AVILABLE = False
    print("Cython module not found")
# Standard Curvelet setups usually use 8 wedges per quadrant at the 2nd coarse scale
DEFAULT_WEDGES = 4

class CurveletFrequencyGrid():
    """
    Class which handles geometric properties and transformations for the Fast Discrete Curvelet Transform (FDCT)
    """
    def __init__(self, nrows: int, ncols: int, scales: int):
        """
        Initialize the Grid.
        Pre-computes the coordinate systems (X, Y, R, Slopes) needed for the masks.

        INPUTS:
            N: int, image size (must be square, e.g. 512)
            nrows: int, image row dimensions
            ncols: int, image column dimensions
            scales: int, total number of scales (including the low-pass center)
        """
        self.nrows = nrows
        self.ncols = ncols
        self.N = max(nrows, ncols)
        self.scales = scales
        
        # Coordinate Grid (Use float to avoid integer division issues)
        # We use a slight offset or 'eps' to avoid division by zero errors in Slopes
        r_rng = np.arange(-self.nrows // 2, self.nrows // 2)
        c_rng = np.arange(-self.ncols // 2, self.ncols // 2)

        # Pass rows FIRST, cols SECOND
        self.Y, self.X = np.meshgrid(r_rng, c_rng, indexing='ij')
        self.Y = self.Y.astype(float)
        self.X = self.X.astype(float)
        # Add epsilon to avoid divide-by-zero (inf is okay, but nan is annoying)
        self.X[self.X == 0] = 1e-10 
        self.Y[self.Y == 0] = 1e-10

        # Slopes
        self.Slopes_EW = self.Y / self.X
        self.Slopes_NS = self.X / self.Y

        self._wedge_cache = {} # Allowes storage of recomputed values

        # Quadrant Masks
        self.Quadrants = {
            "East":  (self.X > 0) & (np.abs(self.Y) <= self.X),
            "West":  (self.X < 0) & (np.abs(self.Y) <= np.abs(self.X)),
            "North": (self.Y < 0) & (np.abs(self.X) <= np.abs(self.Y)),
            "South": (self.Y > 0) & (np.abs(self.X) <= self.Y)
        }

        self.partition_map = self._build_partition_map()




    def _get_scale_bounds(self, scale_idx: int):
        """
        Returns the integer radius boundaries rectangular limits (height_bounds, width_bounds).
        
        INPUTS:
            scale_idx: int, which scale to measure

        RETURNS:
            bounds_rows: Tuple(y_inner, y_outer)
            bounds_cols: Tuple(x_inner, x_outer)
        """
        
        # Inverse logic: Scale 0 is coarsest, Scale (scales-1) is finest
        inverse_scale = (self.scales - 1) - scale_idx
        
        # Vertical row boundaries
        row_outer = self.nrows // (2 ** (inverse_scale + 1))
        
        # Inner boundary (which is the outer boundary of the previous scale)
        # If scale_idx is 0, inner is 0.
        if scale_idx == 0:
            row_inner = 0
        else:
            row_inner = self.nrows // (2 ** (inverse_scale + 2))
        
        bounds_rows = max(1, int(row_inner)), max(1, int(row_outer))

        # Horizontal column boundaries
        col_outer = self.ncols // (2 ** (inverse_scale + 1))

        # Inner boundary
        if scale_idx == 0:
            col_inner = 0
        else:
            col_inner = self.ncols // (2 ** (inverse_scale + 2))
        
        bounds_cols = max(1, int(col_inner)), max(1, int(col_outer))

        return bounds_rows, bounds_cols
    
    def get_radial_window(self, scale_idx: int):
        """
        Generate the Radial 'Donut' Mask.
        Uses Difference of Squares: sqrt(Phi_outer^2 - Phi_inner^2)

        INPUTS:
            scale_idx: int, value of the scale which you'd like mask

        RETURNS:
            mask: 2D array, the circular ring mask
        """
        (r_inner_y, r_outer_y), (r_inner_x, r_outer_x) = self._get_scale_bounds(scale_idx)

        # Outer magnitude
        # The creates the rectangular shell shape (L-infinity norm)
        M_outer = np.maximum(np.abs(self.Y) / r_outer_y, np.abs(self.X) / r_outer_x)
        
        # Coarsest scale, Low-Pass (Phi)
        if scale_idx == 0:
            # Coarsest scale is just the low-pass itself
            return windows.meyer_phi(M_outer)
        
        # Inner magnitude
        M_inner = np.maximum(np.abs(self.Y) / r_inner_y, np.abs(self.X) / r_inner_x)

        # Windows
        if scale_idx == self.scales - 1:
            phi_outer = np.ones_like(self.Y, dtype=float)
        else: 
            # We want the window to be 0 inside r_inner.
            # Phi(R/r_inner) is 1 inside r_inner.
            phi_outer = windows.meyer_phi(M_outer)
        
        phi_inner = windows.meyer_phi(M_inner)
            
        # The "Shell" is the region between them.
        mask = np.sqrt(np.maximum(0, phi_outer**2 - phi_inner**2)) 
        return mask
        
    def get_angular_window(self, quadrant_name, slope_min, slope_max):
        """
        Generate the Angular 'Wedge' Mask.
        Applies meyer_v centered on the wedge.

        INPUTS:
            quadrant_name: str, 'East', 'West', 'North', or 'South'
            slope_min: float, starting slope of the wedge
            slope_max: float, ending slope of the wedge

        RETURNS:
            mask: 2D array, the angular beam mask
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
    
    def _num_wedges_in_scale(self, scale_idx: int) -> int:
        """
        How many wedges exist at this scale.
        
        INPUTS:
            scale_idx: int, the scale to check

        RETURNS:
            count: int, total number of wedges
        """
        if scale_idx == 0:
            return 1
        boundaries = self._get_wedge_slope_ranges(scale_idx)
        wedges_per_quadrant = len(boundaries) - 1
        count = 4 * wedges_per_quadrant
        return count

    def _build_partition_map(self) -> np.ndarray:
        """
        Computes the sum of squares of all filters.
        Used to normalize the inverse transform so energy is preserved.

        INPUTS:
            None

        RETURNS:
            P: 2D array, the normalization grid
        """
        P = np.zeros((self.nrows, self.ncols), dtype=float)

        for j in range(self.scales):
            for w in range(self._num_wedges_in_scale(j)):
                m = self.get_wedge_filter(j, w).astype(float)
                P += m * m

        # avoid divide-by-zero in places your tiling leaves empty
        P[P < 1e-12] = 1.0
        assert np.all(P > 0) # Partition map has zeros; transform is not invertible.
        return P
    
    def get_wedge_filter(self, scale_idx, wedge_idx_in_scale):
        """
        Returns the Soft Wedge Filter for a specific scale and wedge index.
        Combines Radial Donut + Angular Beam + Quadrant Mask.
        Cached for performance.

        INPUTS:
            scale_idx: int, scale index
            wedge_idx_in_scale: int, global wedge index

        RETURNS:
            mask: 2D array, the final isolator filter
        """
        # 0. Check Cache
        cache_key = (scale_idx, wedge_idx_in_scale)
        if cache_key in self._wedge_cache:
            return self._wedge_cache[cache_key]
        
        # 1. Get Radial Donut
        radial_mask = self.get_radial_window(scale_idx)
        
        # If scale 0 (Center), just return the radial mask
        if scale_idx == 0:
            self._wedge_cache[cache_key] = radial_mask
            return radial_mask

        # 2. Determine Wedge Slope Bounds
        boundaries = self._get_wedge_slope_ranges(scale_idx)
        wedges_per_quadrant = len(boundaries) - 1
        
        # 3. Map Global Index to Quadrant & Slope
        quad_names = ["East", "North", "West", "South"]
        quad_idx = wedge_idx_in_scale // wedges_per_quadrant
        slope_idx = wedge_idx_in_scale % wedges_per_quadrant
        
        if quad_idx >= 4:
            raise ValueError(f"Wedge Index {wedge_idx_in_scale} exceeds max for Scale {scale_idx}")

        quadrant = quad_names[quad_idx]
        
        # 4. Get Slope Range
        s_min = boundaries[slope_idx]
        s_max = boundaries[slope_idx+1]
        
        # 5. Get Angular Beam
        angular_mask = self.get_angular_window(quadrant, s_min, s_max)
        
        # 6. Apply Broad Quadrant Mask
        # We allow the wedge to extend naturally past the diagonal.
        # We only cut it off if it wraps all the way around to the opposite side.
        
        if quadrant == "East":
            broad_mask = (self.X > 0)
        elif quadrant == "West":
            broad_mask = (self.X < 0)
        elif quadrant == "North":
            broad_mask = (self.Y < 0) # Numpy Y is positive downwards
        elif quadrant == "South":
            broad_mask = (self.Y > 0)
            
        mask = radial_mask * angular_mask * broad_mask
        # Store mask in cache and return
        self._wedge_cache[cache_key] = mask
        return mask
    
    def _get_wedge_slope_ranges(self, scale_idx: int):
        """
        Calculates the slope boundaries for wedges at a specific scale.
        
        INPUTS:
            scale_idx: int, scale index

        RETURNS:
            boundaries: array, list of slope values [-1, ... , 1]
        """
        if scale_idx == 0: 
            return None
        steps = int((scale_idx - 1) // 2) 
        num_wedges = DEFAULT_WEDGES * (2 ** steps)
        boundaries = np.linspace(-1.0, 1.0, int(num_wedges) + 1)
        return boundaries
    
    def get_wedge_dimensions(self, scale_idx, base_len=None):
        """
        Returns optimal (L1, L2) rectangle size for a wedge at this scale.
        L1 is 'Length' (radial), L2 is 'Width' (Angular).
        
        INPUTS:
            scale_idx: int, scale index

        RETURNS:
            L1: int, radial length
            L2: int, angular width
        """
        if scale_idx == 0:
            # Coarse scale is just a square in the center
            # Pad slightly for safety
            (r_inner_y, r_outer_y), (r_inner_x, r_outer_x) = self._get_scale_bounds(0)
            assert r_outer_y >= 2 and r_outer_x >=2, "Lowpass outer radius too small—check scale bounds."
            dimension_rows= (r_outer_y * 2) + 1
            dimension_cols = (r_outer_x * 2) + 1
            return int(dimension_rows), int(dimension_cols)
        
        if base_len is None:
            base_len = self.N

        # We use parabolic scaling for finer scales
        inverse_scale_idx = (self.scales - 1) - scale_idx

        # Dimensions derived from Candès et al. 2005
        L1 = 4 * base_len // (2 ** (inverse_scale_idx + 2))
        L2 = base_len // (2 ** (inverse_scale_idx//2 + 1)) # Parabolic scaling

        return int(L1), int(L2)
    
    def _get_wedge_center(self, scale_idx, wedge_idx):
        """
        Calculates the geometric center of the wedge using the MASK.
        Used to figure out how much to shift the data before wrapping.

        INPUTS:
            scale_idx: int, scale index
            wedge_idx: int, wedge index

        RETURNS:
            cy: int, center Y coordinate
            cx: int, center X coordinate
        """
        mask = self.get_wedge_filter(scale_idx, wedge_idx)
        grid_y, grid_x = np.indices(mask.shape)
        total_mass = np.sum(mask)
        if total_mass == 0:
            return self.N // 2, self.N // 2
        center_y = np.sum(grid_y * mask) / total_mass
        center_x = np.sum(grid_x * mask) / total_mass
        return int(round(center_y)), int(round(center_x))
    
    def wrap_wedge(self, wedge_data, scale_idx, wedge_idx):
        """
        Cuts out the 'glowing trapezoid' and wraps it into a small rectangle.
        This exploits the periodicity of the FFT.
        Uses Cython optimized code if available.

        INPUTS:
            wedge_data: 2D array, frequency data masked for one wedge
            scale_idx: int, scale index
            wedge_idx: int, wedge index

        RETURNS:
            small_wedge: 2D array, the compact wrapped data (L1 x L2)
        """
        # Determine which quadrant
        boundaries = self._get_wedge_slope_ranges(scale_idx)
        wedges_per_quadrant = len(boundaries) - 1
        quadrant_idx = wedge_idx // wedges_per_quadrant
        quadrant_names = ["East", "North", "West", "South"]
        quadrant = quadrant_names[quadrant_idx]
        
        if quadrant in ["East", "West"]:
            # Horizontal wedges scale with the Column dimension (Width)
            base_len = self.ncols
            L1, L2 = self.get_wedge_dimensions(scale_idx, base_len)
            nrows, ncols = L2, L1 # Swap for horizontal
        else:
            # Vertical wedges scale with the Row dimension (Height)
            base_len = self.nrows
            L1, L2 = self.get_wedge_dimensions(scale_idx, base_len)
            nrows, ncols = L1, L2

        # Find the approximate center of the wedge (to be changed later)
        cy, cx = self._get_wedge_center(scale_idx, wedge_idx)
        
        # Sanitize cy and cx for cython
        cy = (cy % self.nrows) + self.nrows
        cx = (cx % self.ncols) + self.ncols

        # Use optimized code if available
        if CYTHON_AVILABLE:
            return inner_loop.wrap_wedge_fast(wedge_data, nrows, ncols, cy, cx, self.nrows, self.ncols)

        # Default to python based implementation if Cython code not available
        # Cut out rectangle, handle indices moved by np.roll
        shift_x_center = (self.ncols // 2) - cx
        shift_y_center = (self.nrows // 2) - cy

        centered_data = np.roll(wedge_data, shift_y_center, axis=0)
        centered_data = np.roll(centered_data, shift_x_center, axis=1)

        # Slice middle pixels
        start_x = (self.ncols // 2) - (ncols // 2)
        start_y = (self.nrows // 2) - (nrows // 2)

        small_wedge = centered_data[start_y:start_y + nrows, start_x:start_x + ncols]

        return small_wedge
    
    def unwrap_wedge(self, wrapped_data, scale_idx, wedge_idx):
        """
        Reverses the wrapping.
        Puts the small L1xL2 wedge back into the big N x N grid at the correct position.
        Uses Cython optimized code if available.

        INPUTS:
            wrapped_data: 2D array, compact coefficient data
            scale_idx: int, scale index
            wedge_idx: int, wedge index

        RETURNS:
            unwrapped_grid: 2D array, full size grid with wedge placed correctly
        """
        nrows, ncols = wrapped_data.shape
        
        cy, cx = self._get_wedge_center(scale_idx, wedge_idx)

        # sanitze cx and cy for cython
        cy = (cy % self.nrows) + self.nrows
        cx = (cx % self.ncols) + self.ncols

        # Fast path, if Cython available
        if CYTHON_AVILABLE:
            # Create a blank grid and let C accumulate the wedge onto it
            # Handles complex vs float via fused types
            big_grid = np.zeros((self.nrows, self.ncols), dtype=wrapped_data.dtype)

            # Modify big grid in place
            inner_loop.unwrap_wedge_fast(wrapped_data, big_grid, cy, cx, self.nrows, self.ncols)
            return big_grid
        
        # Default to python implementation if cython not available

        # Create the target grid
        big_grid = np.zeros((self.nrows, self.ncols), dtype=complex)

        # Place small wedge in center of grid
        start_y = (self.nrows // 2) - (nrows // 2)
        start_x = (self.ncols // 2) - (ncols // 2)

        big_grid[start_y:start_y + nrows, start_x:start_x + ncols] = wrapped_data

        # Determine shift
        shift_y_center = (self.nrows // 2) - cy
        shift_x_center = (self.ncols // 2) - cx

        # Unroll
        unwrapped_grid = np.roll(big_grid, -shift_y_center, axis=0)
        unwrapped_grid = np.roll(unwrapped_grid, -shift_x_center, axis=1)

        return unwrapped_grid
    

    def forward_transform(self, image):
        """
        Perform Fast Discrete Curvelet Transform via Wrapping.
        
        INPUTS:
            image: 2D array, input image (spatial domain)

        RETURNS:
            coefficients: list of lists, curvelet coefficients organized by [scale][wedge]
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

                dim_rows, dim_cols = self.get_wedge_dimensions(0) # Returns L x L for coarse rectangle

                # Cut out center (image already centered so crop is simple)
                center_y, center_x = self.nrows // 2, self.ncols // 2
                rad_y = dim_rows // 2
                rad_x = dim_cols // 2

                # Get slice indices
                s_row = slice(center_y - rad_y, center_y + rad_y + 1)
                s_col = slice(center_x - rad_x, center_x + rad_x+ 1)

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
        Performs Inverse Fast Discrete Curvelet Transform via Wrapping.
        
        INPUTS:
            coefficients: list of lists, curvelet coefficients [scale][wedge]
        
        RETURNS:
            reconstructed_image: 2D array, the restored image
        """

        reconstructed_frequency = np.zeros((self.nrows, self.ncols), dtype=complex)

        for j, scale_coeffs in enumerate(coefficients):

            if j == 0:
                # FFT to get back to frequency
                data = scale_coeffs[0]
                frequency_data = np.fft.fftshift(np.fft.fft2(data))

                # 1. Get dimensions of the small lowpass grid
                h_small, w_small = frequency_data.shape
                
                # 2. Prepare the big grid
                temporary_grid = np.zeros((self.nrows, self.ncols), dtype=complex)

                # 3. Calculate Centers
                cy, cx = self.nrows // 2, self.ncols // 2
                ry, rx = h_small // 2, w_small // 2

                # 4. Define Slices (Center of big grid)
                s_row = slice(cy - ry, cy + ry + 1)
                s_col = slice(cx - rx, cx + rx + 1)
                
                # 5. Paste safely
                # frequency_data is already centered, so we copy it whole.
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
        reconstructed_frequency = reconstructed_frequency / self.partition_map

        reconstructed_image = np.fft.ifft2(np.fft.ifftshift(reconstructed_frequency))

        return np.real(reconstructed_image)


