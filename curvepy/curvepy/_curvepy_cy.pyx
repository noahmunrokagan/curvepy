# _curvelet_cy.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, pi, sqrt, abs, pow, floor

# Define standard complex type
ctypedef double complex c128

# ---------------------------------------------------------
# 1. C-Level Math Helpers (The Math Kernel)
# ---------------------------------------------------------

cdef inline double meyer_auxiliary_c(double x) nogil:
    """Polynomial smoother: 35x^4 - 84x^5 + 70x^6 - 20x^7"""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    return 35.0*pow(x, 4) - 84.0*pow(x, 5) + 70.0*pow(x, 6) - 20.0*pow(x, 7)

cdef inline double meyer_phi_c(double omega) nogil:
    """Low-pass window function"""
    omega = abs(omega)
    if omega <= 0.5:
        return 1.0
    elif omega >= 1.0:
        return 0.0
    
    # Transition [0.5, 1.0] -> Map to [0, 1] for auxiliary
    # val goes from 0 (at 0.5) to 1 (at 1.0)
    cdef double val = (omega - 0.5) * 2.0 
    
    # We want 1 -> 0, so we plug (1-val) into auxiliary
    return sin((pi / 2.0) * meyer_auxiliary_c(1.0 - val))

cdef inline double meyer_v_c(double t) nogil:
    """Angular window function"""
    t = abs(t)
    if t >= 1.0:
        return 0.0
    return sin((pi / 2.0) * meyer_auxiliary_c(1.0 - t))

# ---------------------------------------------------------
# 2. Main Processing Kernel
# ---------------------------------------------------------

def build_wedge_and_wrap(
    c128[:, :] freq_image,      # The full N x N frequency data
    int N,                      # Image dimension
    int scale_idx,              # 0 = Lowpass, >0 = Wedge
    # Radial params
    double r_inner,
    double r_outer,
    # Angular params
    double s_min,
    double s_max,
    int quad_id,                # 0=East, 1=North, 2=West, 3=South
    # Output Geometry
    int L1,                     # Height of output wedge
    int L2,                     # Width of output wedge
    int cy,                     # Center Y of the wedge in N x N
    int cx                      # Center X of the wedge in N x N
):
    """
    Fused operation:
    1. Loop over the SMALL target L1xL2 grid (fast!)
    2. Map target (r,c) -> source (y,x) in N x N grid
    3. Compute Mask value at (y,x) on the fly
    4. Copy * Mask
    """
    
    # Allocate Output (Python Object, handled by Numpy)
    cdef c128[:, :] output = np.zeros((L1, L2), dtype=np.complex128)
    
    # Loop variables
    cdef int r, c
    cdef int src_y, src_x
    
    # Wrapping shifts
    # We want center of L1xL2 to map to (cy, cx) in global
    cdef int start_y = cy - (L1 // 2)
    cdef int start_x = cx - (L2 // 2)

    # Pre-calc slope constants
    cdef double slope_center = (s_min + s_max) * 0.5
    cdef double slope_width = s_max - s_min
    
    # Coordinate vars
    cdef double y_coord, x_coord, radius, slope, norm_slope
    cdef double phi_low, phi_high, rad_mask, ang_mask, total_mask
    cdef double epsilon = 1e-10

    # Parallelize this loop via prange if you setup OpenMP, 
    # but sequential is plenty fast for now.
    for r in range(L1):
        for c in range(L2):
            
            # 1. FIND GLOBAL COORDINATE (With Wrap)
            # The coordinate in the "virtual" unwrapped grid
            src_y = start_y + r
            src_x = start_x + c
            
            # Handle wrapping (modulo N)
            # ((x % N) + N) % N handles negative numbers correctly in C
            src_y = ((src_y % N) + N) % N
            src_x = ((src_x % N) + N) % N
            
            # 2. CALCULATE GEOMETRY (X, Y, R, Slope)
            # Convert index to centered grid coordinates [-N/2, N/2]
            y_coord = <double>(src_y - N//2)
            x_coord = <double>(src_x - N//2)
            
            # Fix Zero for division safety
            if x_coord == 0: x_coord = epsilon
            if y_coord == 0: y_coord = epsilon
            
            # Radius (L-infinity norm for square shells)
            radius = max(abs(x_coord), abs(y_coord))
            
            # 3. COMPUTE RADIAL MASK
            if scale_idx == 0:
                # Lowpass
                rad_mask = meyer_phi_c(radius / r_outer)
                # No angular mask for lowpass
                total_mask = rad_mask
            else:
                # Bandpass
                # Check bounding box to avoid expensive calls (Optimization)
                if radius < r_inner * 0.9 or radius > r_outer * 1.1:
                    total_mask = 0.0
                else:
                    phi_high = meyer_phi_c(radius / r_outer)
                    phi_low  = meyer_phi_c(radius / r_inner)
                    rad_mask = sqrt(max(0.0, phi_high*phi_high - phi_low*phi_low))

                    # 4. COMPUTE ANGULAR MASK
                    if rad_mask == 0.0:
                        total_mask = 0.0
                    else:
                        # Calculate Slope based on Quadrant
                        # East/West: y/x, North/South: x/y
                        if quad_id == 0 or quad_id == 2: # East/West
                            slope = y_coord / x_coord
                        else: # North/South
                            slope = x_coord / y_coord
                        
                        norm_slope = (slope - slope_center) / slope_width
                        ang_mask = meyer_v_c(norm_slope)
                        
                        # 5. QUADRANT CHECK
                        # Ensure we don't pick up the symmetric wedge on the opposite side
                        cdef bint correct_quad = False
                        if quad_id == 0: correct_quad = (x_coord > 0)
                        elif quad_id == 1: correct_quad = (y_coord < 0) # Numpy Y is down? Check your convention. 
                                                                       # Usually Y<0 is 'North' in image indices if 0,0 is top-left
                        elif quad_id == 2: correct_quad = (x_coord < 0)
                        elif quad_id == 3: correct_quad = (y_coord > 0)
                        
                        if not correct_quad:
                            total_mask = 0.0
                        else:
                            total_mask = rad_mask * ang_mask

            # 6. APPLY AND STORE
            if total_mask > 0.0:
                output[r, c] = freq_image[src_y, src_x] * total_mask
                
    return np.asarray(output)