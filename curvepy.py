import numpy as np
import matplotlib.pyplot as plt

# Standard Curvelet setups usually use 8 wedges per quadrant at the 2nd coarse scale
DEFAULT_WEDGES = 4

class CurveletFrequencyGrid():
    def __init__(self, N: int, scales: int):
        self.N = N
        self.scales = scales
        
        # 1. Coordinate Grid (Use float to avoid integer division issues)
        # We use a slight offset or 'eps' to avoid division by zero errors in Slopes
        self.Y, self.X = np.mgrid[-N//2:N//2, -N//2:N//2].astype(float)
        
        # Add epsilon to avoid divide-by-zero (inf is okay, but nan is annoying)
        self.X[self.X == 0] = 1e-10 
        self.Y[self.Y == 0] = 1e-10

        self.Slopes_EW = self.Y / self.X
        self.Slopes_NS = self.X / self.Y

        # 2. Quadrant Masks
        # Note the use of bitwise '&' and parens
        self.Quadrants = {
            "East":  (self.X > 0) & (np.abs(self.Y) <= self.X),
            "West":  (self.X < 0) & (np.abs(self.Y) <= np.abs(self.X)),
            "North": (self.Y < 0) & (np.abs(self.X) <= np.abs(self.Y)),
            "South": (self.Y > 0) & (np.abs(self.X) <= self.Y)
        }

    def _get_corona_indices(self, scale_idx: int):
        """
        Returns the (row_start, row_end), (col_start, col_end) for the square 
        at this scale.
        """
        center_idx = self.N // 2
        
        # Determine how "deep" this scale is from the finest level
        # Scale (scales-1) is the Full Image (Radius N/2)
        # Scale 0 is the Coarsest (Radius N / 2^scales)
        
        # We calculate the side length (radius) of the square for this scale
        # Logic: We work backwards from the border.
        inverse_scale = (self.scales - 1) - scale_idx
        radius = self.N // (2 ** (inverse_scale + 1))
        
        # Ensure at least radius 1
        radius = max(1, int(radius))

        # Convert radius to slice indices centered at N/2
        row_start = center_idx - radius
        row_end   = center_idx + radius
        col_start = center_idx - radius
        col_end   = center_idx + radius

        return (row_start, row_end), (col_start, col_end)

    def _get_corona_mask(self, scale_idx: int) -> np.ndarray:
        mask = np.zeros((self.N, self.N), dtype=bool)

        # 1. Get Outer Square (The current scale boundary)
        r_out, c_out = self._get_corona_indices(scale_idx)
        mask[r_out[0]:r_out[1], c_out[0]:c_out[1]] = True
    
        # 2. Subtract Inner Square (The previous scale) to make a 'Donut'
        # Scale 0 is a solid square, so we don't subtract anything.
        if scale_idx > 0:
            r_in, c_in = self._get_corona_indices(scale_idx - 1)
            mask[r_in[0]:r_in[1], c_in[0]:c_in[1]] = False

        return mask
    
    def _get_wedge_slope_ranges(self, scale_idx: int):
        # Scale 0 is isotropic (no wedges), just a square
        if scale_idx == 0:
            return None

        # Parabolic Scaling: Double the number of wedges every 2 scales
        # scale_idx 1 -> steps 0
        # scale_idx 2 -> steps 0
        # scale_idx 3 -> steps 1
        steps = int((scale_idx - 1) // 2) 
        num_wedges = DEFAULT_WEDGES * (2 ** steps)
        
        # Create slope boundaries from -1 to 1
        return np.linspace(-1.0, 1.0, int(num_wedges) + 1)

    def build_grid(self):
        all_masks = []
        
        for scale in range(self.scales):
            # A. Handle Center (Low Pass)
            if scale == 0:
                all_masks.append(self._get_corona_mask(0))
                continue

            # B. Get the Square Ring
            ring_mask = self._get_corona_mask(scale)
            
            # C. Get Wedge Boundaries
            boundaries = self._get_wedge_slope_ranges(scale)
            
            # D. Iterate Quadrants
            for quadrant in ['East', 'West', 'North', 'South']:
                # FIX: Check list membership properly
                if quadrant in ["East", "West"]:
                    current_slopes = self.Slopes_EW
                else:
                    current_slopes = self.Slopes_NS
                
                # Get the base quadrant mask
                quadrant_mask = self.Quadrants[quadrant]

                # We have N+1 boundaries, which creates N bins (wedges)
                # Loop through the bins
                for i in range(len(boundaries) - 1):
                    slope_low = boundaries[i]
                    slope_high = boundaries[i+1]
                    
                    # Create mask based on slope range
                    # Note: We use >= and < to avoid overlapping pixels
                    slope_mask = (current_slopes >= slope_low) & (current_slopes < slope_high)
                    
                    # Combine: Must be in Ring AND in Quadrant AND in Slope Range
                    # FIX: Use bitwise '&'
                    wedge = ring_mask & quadrant_mask & slope_mask
                    
                    all_masks.append(wedge)

        return all_masks

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