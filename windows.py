"""
Python file which implements the Meyer Window function. If I "hard cut", then ringing/ripples will occur (Gibbs phoenomenon).
Guarantees invertibility and isometry (energy preservation)
"""
from typing import Union
import math 
def meyer_auxiliary(x: Union[int, float]):
    # This polynomial is the standard "Meyer" transition
    # Input x: a value strictly between 0 and 1
    # Output: a value smoothly transitioning from 0 to 1

    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        return 35 * x ** 4 - 84 * x ** 5 + 70 * x ** 6 - 20 * x ** 7


def phi_1d(frequency_val):
    # frequency_val: the coordinate in the frequency domain
    
    # Take absolute value (symmetry around 0)
    val = abs(frequency_val)

    # Define transition boundaries
    # Note: The paper suggests support might vanish outside [-2, 2] [cite: 193]
    # Standard implementations often transition between 1/2 and 1, or 1/2 and 2.
    # Let's assume a transition band [A, B] (e.g., 1/2 to 1)

    lower_bound = 1/2
    upper_bound = 1.0

    if val <= lower_bound:
        return 1.0
    elif val >= upper_bound:
        return 0.0
    else:
        # Map the value into the [0, 1] range for the auxiliary function
        # We want 1 when close to lower_bound and 0 when close to upper_bound
        normalized_pos = (val - lower_bound) / (upper_bound - lower_bound)
        
        # Invert normalized_pos because we are going from 1 down to 0
        argument = 1 - normalized_pos
        
        # Apply the "Smooth Step"
        # Often implemented as sin(pi/2 * meyer_aux(argument)) to ensure
        # that squares sum to 1 (sin^2 + cos^2 = 1).
        return math.sin( (math.pi / 2) * meyer_auxiliary(argument) )