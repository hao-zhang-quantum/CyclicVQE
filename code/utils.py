import matplotlib.pyplot as plt
from pennylane import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

def analyze_state(state, cutoff=1e-10):
    import numpy as onp

    n_total = int(onp.log2(len(state)))
    n_main = n_total - 1  # Remove the last auxiliary bit
    amplitudes = []

    for i, amp in enumerate(state):
        prob = abs(amp)
        if prob > cutoff:
            full_bitstring = format(i, f'0{n_total}b')
            trimmed_bitstring = full_bitstring[:n_main]
            amplitudes.append((prob, trimmed_bitstring, amp))

    # Sort by magnitude
    amplitudes.sort(reverse=True, key=lambda x: x[0])

    print(f"{'Amplitude':>20} {'Trimmed Bitstring':>20} {'Full complex':>30}")
    for prob, bitstring, amp in amplitudes:
        print(f"{prob:20.12f} {bitstring:>20} {str(amp):>30}")


    coeffs = np.array([x[2] for x in amplitudes])
    bases = np.array([[int(b) for b in x[1]] for x in amplitudes], dtype=int)


    return coeffs, bases

def pretty_print_state(coeffs, bases, cutoff=1e-10, decimal=4):
    """
    Print quantum state in Dirac notation like:
    +0.7071 |101⟩
    -0.5000j |011⟩
    """
    from numpy import abs as np_abs
    from numpy import angle
    from numpy import pi

    for coeff, base in zip(coeffs, bases):
        if np_abs(coeff) < cutoff:
            continue
        bitstring = ''.join(map(str, base))
        real = coeff.real
        imag = coeff.imag

        # Format depending on whether it's real, imag, or complex
        if abs(real) < cutoff and abs(imag) >= cutoff:
            coeff_str = f"{imag:.{decimal}f}j"
        elif abs(imag) < cutoff:
            coeff_str = f"{real:+.{decimal}f}"
        else:
            coeff_str = f"{real:+.{decimal}f}{imag:+.{decimal}f}j"

        print(f"{coeff_str} |{bitstring}⟩")



def plot_orbital_occupation_map(
    coeffs, bases,
    figscale = 1,
    arrow_scale=0.3, cmap='hsv',
    order_by_decimal=True
):
    """
    - Colorbar placed horizontally at the bottom and shrunk
    - If order_by_decimal=True, sort by decimal value of bitstring in descending order (HF state automatically at the bottom)
    """
    n_qubits  = bases.shape[1]
    n_states  = len(bases)

    mags        = np.abs(coeffs)
    phases      = np.angle(coeffs)
    norm_phases = (phases + 2*np.pi) % (2*np.pi) / (2*np.pi)

    if order_by_decimal:
        # Calculate decimal value for each basis state
        decimals = [int(''.join(map(str, row)), 2) for row in bases]
        # Sort in descending order: maximum (HF) automatically at idx=0
        sorted_idx = sorted(range(n_states),
                            key=lambda i: decimals[i],
                            reverse=True)
        bases       = bases[sorted_idx]
        mags        = mags[sorted_idx]
        phases      = phases[sorted_idx]
        norm_phases = np.array(norm_phases)[sorted_idx]
        n_states    = len(bases)

    # Prepare for plotting
    norm    = Normalize(0,1)
    cmap_obj = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=(1.4*n_qubits*figscale, 0.6*n_states*figscale))

    for i, (bitrow, amp, phase, nph) in enumerate(zip(
        bases, mags, phases, norm_phases
    )):
        color = cmap_obj(norm(nph))
        for j, bit in enumerate(bitrow):
            if bit:
                ax.scatter(j, i,
                           s=600*amp,
                           color=color,
                           edgecolor='black', alpha=0.9)
                # Total arrow length L includes head
                L  = arrow_scale * amp
                hl = 0.3 * L   # Head length = 30%
                hw = 0.3 * L   # Head width = 30%
                sw = 0.02      # Shaft width (optional: amp * sw)

                dx = L * np.cos(phase)
                dy = L * np.sin(phase)
                ax.arrow(j, i, dx, dy,
                         width=sw,
                         head_width=hw,
                         head_length=hl,
                         length_includes_head=True,
                         color='black', alpha=0.6)

    # Coordinates and grid
    ax.set_xticks(range(n_qubits))
    ax.set_xticklabels([f"orb {k}" for k in range(n_qubits)])
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f"|{''.join(map(str,b))}⟩" for b in bases])
    ax.set_xlim(-0.5, n_qubits - 0.5)
    ax.set_ylim(-1,    n_states)
    ax.set_title("Orbital Occupation", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Horizontal small colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    fig.colorbar(
        sm, ax=ax,
        orientation='horizontal',
        pad=0.12,
        shrink=0.6,
        aspect=20,
        label="Phase / 2π",
    )

    plt.tight_layout()
    plt.show()



def plot_orbital_occupation_maps(
    coeffs_bases_list, titles=None,
    figscale=1, arrow_scale=0.3, cmap='hsv',
    order_by_decimal=True
):
    """
    Plot multiple orbital occupation maps for side-by-side comparison.
    
    Parameters:
    - coeffs_bases_list: List, each element is a (coeffs, bases) pair
    - titles: Optional list of subplot titles
    - figscale: Scaling factor
    - arrow_scale: Arrow length baseline
    - cmap: Colormap
    - order_by_decimal: Whether to sort by decimal value of bitstring in descending order
    """
    n_plots = len(coeffs_bases_list)
    if titles is None:
        titles = [f"Data {i+1}" for i in range(n_plots)]

    # Get number of qubits and states from the first data
    n_qubits = coeffs_bases_list[0][1].shape[1]
    n_states = len(coeffs_bases_list[0][1])
    
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(1.4 * n_qubits * figscale * n_plots, 0.6 * n_states * figscale),
        squeeze=False
    )
    axes = axes[0]

    norm = Normalize(0, 1)
    cmap_obj = plt.get_cmap(cmap)
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    for ax, (coeffs, bases), title in zip(axes, coeffs_bases_list, titles):
        mags = np.abs(coeffs)
        phases = np.angle(coeffs)
        norm_phases = ((phases + 2 * np.pi) % (2 * np.pi)) / (2 * np.pi)

        if order_by_decimal:
            decimals = [int(''.join(map(str, row)), 2) for row in bases]
            sorted_idx = sorted(range(len(bases)), key=lambda i: decimals[i], reverse=True)
            bases = bases[sorted_idx]
            mags = mags[sorted_idx]
            phases = phases[sorted_idx]
            norm_phases = norm_phases[sorted_idx]

        for i, (bitrow, amp, phase, nph) in enumerate(zip(bases, mags, phases, norm_phases)):
            color = cmap_obj(norm(nph))
            for j, bit in enumerate(bitrow):
                if bit:
                    ax.scatter(j, i, s=600 * amp, color=color, edgecolor='black', alpha=0.9)
                    L = arrow_scale * amp
                    hl = 0.3 * L  # Head length
                    hw = 0.3 * L  # Head width
                    sw = 0.02     # Shaft width
                    dx = L * np.cos(phase)
                    dy = L * np.sin(phase)
                    ax.arrow(j, i, dx, dy,
                             width=sw, head_width=hw, head_length=hl,
                             length_includes_head=True, color='black', alpha=0.6)

        ax.set_xticks(range(n_qubits))
        ax.set_xticklabels([f"orb {k}" for k in range(n_qubits)])
        ax.set_yticks(range(len(bases)))
        ax.set_yticklabels([f"|{''.join(map(str, b))}⟩" for b in bases])
        ax.set_xlim(-0.5, n_qubits - 0.5)
        ax.set_ylim(-1, len(bases))
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)



    cbar_ax = fig.add_axes([0.1, -0.05, 0.8, 0.03])  # Left, bottom, width, height
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label="Phase / 2π")

    plt.tight_layout()
    plt.show()


def ensure_file_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)