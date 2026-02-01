import time
import cProfile
import pstats
import numpy as np
from curvepy.curvepy import CurveletFrequencyGrid

def benchmark():
    # Use a realistic size (512x512) and enough scales to stress the system
    N = 512
    fdct = CurveletFrequencyGrid(N=N, scales=5)
    img = np.random.rand(N, N)
    
    print(f"Running Benchmark on {N}x{N} image...")
    
    # Measure Forward
    start = time.time()
    coeffs = fdct.forward_transform(img)
    fwd_time = time.time() - start
    print(f"Forward Transform: {fwd_time:.4f} seconds")

    # Measure Inverse
    start = time.time()
    recon = fdct.inverse_transform(coeffs)
    inv_time = time.time() - start
    print(f"Inverse Transform: {inv_time:.4f} seconds")

if __name__ == "__main__":
    # 1. Run raw timing
    benchmark()
    
    # 2. Run detailed breakdown
    print("\n--- DETAILED PROFILING ---")
    profiler = cProfile.Profile()
    profiler.enable()
    benchmark()
    profiler.disable()
    
    # Sort by 'cumulative time' to see the biggest offenders
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(15)