"""
read_bsf_gnu.py  —  Load and plot a SPR-KKR Bloch spectral function from a .gnu file.

Usage:
    python read_bsf_gnu.py <path/to/file.gnu>

Example:
    python read_bsf_gnu.py ../../BLOCHSF/BSF/A1_FM_eta0.0/HEA_FM_A1_eta0.0_BLOCHSF_tot_bsf.gnu

The .gnu file is a 3-column ASCII output (k-path, energy [Ry], intensity) written by
SPR-KKR alongside the binary .bsf file. Columns: k_path [2pi/a], E [Ry], A(k,E) [a.u.].

Output: <input_stem>_plot.pdf saved next to the input file.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ── Band structure plot defaults (edit as needed) ──────────────────────────────
TITLE        = ''
CMAP         = 'gnuplot2'
VMIN         = 0
VMAX         = 400
Y_LIM        = (-10, 5)      # eV
RY_TO_EV     = 13.605693123

# FCC k-path tick positions and labels (cumulative distance in 2pi/a units).
# These match the default KPATH=2 FCC path in SPR-KKR (X-Γ-L-W-K-Γ).
# Adjust if using a different path or phase.
K_TICKS_POS  = [0.0000, 1.0000, 1.8660, 2.5731, 2.9267, 3.9874]
K_TICK_LABS  = ['X', r'$\Gamma$', 'L', 'W', 'K', r'$\Gamma$']


def load_gnu(path):
    """Read 3-column .gnu file; return (K, E, I) 2D grids (NK × NE)."""
    data = np.genfromtxt(path, dtype=np.float64)
    k_f, e_f, i_f = data[:, 0], data[:, 1], data[:, 2]

    # Detect number of energy points from the first k-value repetition
    first_k  = k_f[0]
    changes  = np.where(k_f != first_k)[0]
    NE = int(changes[0]) if len(changes) else len(k_f)
    NK = len(k_f) // NE
    n  = NK * NE

    K = k_f[:n].reshape(NK, NE)
    E = e_f[:n].reshape(NK, NE)
    I = i_f[:n].reshape(NK, NE)
    print(f'Grid: NK={NK}, NE={NE}  |  I range: [{I.min():.3g}, {I.max():.3g}]')
    return K, E, I


def plot_bsf(K, E, I, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))

    E_eV = E * RY_TO_EV
    ax.pcolormesh(K, E_eV, I, cmap=CMAP, vmin=VMIN, vmax=VMAX, shading='auto')

    ax.set_ylim(Y_LIM)
    ax.set_ylabel(r'$E - E_{\rm F}$ (eV)')
    ax.axhline(0, color='white', lw=0.5, ls='--')

    k_max   = K.max()
    k_scale = k_max / max(K_TICKS_POS)
    ticks   = [p * k_scale for p in K_TICKS_POS]
    ax.set_xlim(0, k_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels(K_TICK_LABS)
    for xp in ticks:
        ax.axvline(xp, color='white', lw=0.4, alpha=0.6)

    if TITLE:
        ax.set_title(TITLE)

    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Plot SPR-KKR BSF from a .gnu file.')
    ap.add_argument('gnu_file', help='Path to the .gnu band-structure file')
    args = ap.parse_args()

    path    = Path(args.gnu_file)
    K, E, I = load_gnu(path)
    out     = path.with_suffix('').with_suffix('') if path.name.endswith('.gnu') else path
    plot_bsf(K, E, I, path.parent / (path.stem + '_plot.pdf'))


if __name__ == '__main__':
    main()
