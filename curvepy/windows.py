"""
windows.py
Implements the Meyer Window functions for the Fast Discrete Curvelet Transform.
"""
import numpy as np
import matplotlib.pyplot as plt

def meyer_auxiliary(x):
    """
    The polynomial h(x) typically used in Meyer wavelets.
    Smoothly transitions from 0 to 1 on the interval [0, 1].
    Formula: 35x^4 - 84x^5 + 70x^6 - 20x^7
    """
    # Clamp inputs to [0, 1] for the polynomial calculation
    x_clamped = np.clip(x, 0, 1)
    return 35 * x_clamped**4 - 84 * x_clamped**5 + 70 * x_clamped**6 - 20 * x_clamped**7

def meyer_phi(omega):
    """
    The Low-Pass Scaling Function Phi(omega).
    1 inside the low-frequency core, 0 outside.
    Support: typically decays from 1 to 0 in the interval [1/2, 1].
    """
    omega = np.abs(omega)
    
    # Initialize output array
    phi = np.zeros_like(omega)
    
    # 1. Region strictly inside lower bound (Pure 1.0)
    phi[omega <= 0.5] = 1.0
    
    # 2. Transition Region (0.5 < omega < 1.0)
    # We want to go from 1 down to 0.
    # We map [0.5, 1.0] to [0, 1] for the auxiliary function.
    mask_transition = (omega > 0.5) & (omega < 1.0)
    
    # Normalized argument: 0 at 0.5, 1 at 1.0
    val = (omega[mask_transition] - 0.5) / 0.5 
    
    # Apply Smooth Step: sin(pi/2 * h(x))
    # Note: We use (1 - val) because we want to start at 1 and drop to 0
    phi[mask_transition] = np.sin((np.pi / 2) * meyer_auxiliary(1 - val))
    
    # 3. Region strictly outside (Pure 0.0) is already handled by np.zeros
    return phi

def meyer_psi(omega):
    """
    The Band-Pass Wavelet Function Psi(omega).
    Isolates a specific 'donut' ring in frequency.
    Psi(w) = sqrt(Phi(w/2)^2 - Phi(w)^2)
    """
    # This ensures that |Phi|^2 + |Psi|^2 + ... sums to 1 (Energy preservation)
    return np.sqrt(meyer_phi(omega / 2)**2 - meyer_phi(omega)**2)

def meyer_v(t):
    """
    The Angular Window V(t).
    Used for smooth wedge transitions.
    Typically supported on [-1, 1].
    """
    # V(t) must satisfy V(t)^2 + V(t-1)^2 = 1 for the partition of unity.
    
    t = np.array(t)
    v = np.zeros_like(t)
    
    mask = np.abs(t) <= 1
    
    val = np.abs(t[mask]) # 0 to 1
    v[mask] = np.sin((np.pi / 2) * meyer_auxiliary(1 - val))
    
    return v


if __name__ == "__main__":
    print("Plotting Meyer Windows...")
    
    # 1. Setup Domain
    x = np.linspace(0, 2.5, 500)
    t = np.linspace(-1.5, 1.5, 500)
    
    # 2. Compute Functions
    y_phi = meyer_phi(x)
    y_psi = meyer_psi(x)
    y_v   = meyer_v(t)
    
    # 3. Plot
    plt.figure(figsize=(12, 5))
    
    # Plot Radial Components
    plt.subplot(1, 2, 1)
    plt.plot(x, y_phi, 'b-', linewidth=2, label=r'Low Pass $\Phi(\omega)$')
    plt.plot(x, y_psi, 'r--', linewidth=2, label=r'Band Pass $\Psi(\omega)$')
    plt.title("Radial Meyer Windows")
    plt.xlabel(r"Frequency $\omega$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axvline(0.5, color='k', linestyle=':', alpha=0.3)
    plt.axvline(1.0, color='k', linestyle=':', alpha=0.3)
    plt.axvline(2.0, color='k', linestyle=':', alpha=0.3)

    # Plot Angular Component
    plt.subplot(1, 2, 2)
    plt.plot(t, y_v, 'g-', linewidth=2, label=r'Angular Window $V(t)$')
    plt.title("Angular Meyer Window")
    plt.xlabel("Slope / Angle")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    