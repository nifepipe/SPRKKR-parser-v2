"""
Imports and processes data from SPRKKR outputs for FSURF calculations
Extracts and plots a cut through the Fermi Surface calculated with SPR-KKR and XBAND

Author: David Redka, Nicolas Piwek
Date: 2025.12.09
"""
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration Constants --- FCC LATTICE!!!
BZ_VERTICES = np.array([
    [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5],  # Outer square
    [0.0, 0.0],  # Center (Γ point)
    [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0], [0.0, -0.5],    # Midpoints of edges
])

BZ_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Outer square
    (0, 4), (1, 4), (2, 4), (3, 4),  # Diagonals to center (Γ)
    (5, 6), (6, 7), (7, 8), (8, 5)   # Cross edges
]

SYMMETRY_POINTS = {
    "Γ": (0.0, 0.0),
    "M": (0.5, 0.5),
    "X": (0.5, 0.0),
    "M$_{x}$": (-0.5, 0.5),
    "M$_{y}$": (0.5, -0.5),
    "M$_{xy}$": (-0.5, -0.5),
    "X$_{x}$": (-0.5, -0.0),
}


def select_file() -> Optional[Path]:
    """
    Opens a file dialog to allow the user to select a data file.
    Returns the Path object if selected, else None.
    """
    root = tk.Tk()
    root.withdraw()
    
    file_filters = [
        ("GNU PlotFiles", "*.gnu"),  # Shows files ending in .gnu
        ("All Files", "*.*")        # Default option to show all files
    ]

    file_path_str = filedialog.askopenfilename(
        initialdir=".", 
        title="Select a file",
        filetypes=file_filters

    )
    
    root.destroy()

    if file_path_str:
        return Path(file_path_str)
    return None


def load_and_reshape_data(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads 3-column data from a file and reshapes it into 2D grids.
    Assumes the data represents a square grid.
    """
    data = np.genfromtxt(file_path)
    x_raw, y_raw, z_raw = data[:, 0], data[:, 1], data[:, 2]

    # Calculate grid dimension N assuming a square grid
    n_points = np.size(x_raw)
    n_dim = int(np.sqrt(n_points))
    if n_dim * n_dim != n_points:
        raise ValueError(f"Data size {n_points} is not a perfect square. Cannot reshape.")

    # Reshape
    xx = np.reshape(x_raw, (n_dim, n_dim))
    yy = np.reshape(y_raw, (n_dim, n_dim))
    zz = np.reshape(z_raw, (n_dim, n_dim))

    return xx, yy, zz


def symmetrize_data(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies mirroring/symmetry operations to expand the grid.
    1. Mirrors along Y-axis (vertical flip).
    2. Mirrors along X-axis (horizontal flip).
    """
    
    # --- Mirror along Y-axis (Vertical) ---
    zz_fly = np.flipud(zz)
    yy_fly = yy 
    xx_fly = -np.flipud(xx)

    xx = np.concatenate([xx_fly, xx], axis=0)
    yy = np.concatenate([yy_fly, yy], axis=0)
    zz = np.concatenate([zz_fly, zz], axis=0)

    # --- Mirror along X-axis (Horizontal) ---
    zz_fly = np.fliplr(zz)
    yy_fly = -np.fliplr(yy)
    xx_fly = xx

    xx = np.concatenate([xx_fly, xx], axis=1)
    yy = np.concatenate([yy_fly, yy], axis=1)
    zz = np.concatenate([zz_fly, zz], axis=1)

    return xx, yy, zz

def add_symmetry_points(plt)->None:
    # Draw Brillouin Zone (BZ)
    for edge in BZ_EDGES:
        p1 = BZ_VERTICES[edge[0]]
        p2 = BZ_VERTICES[edge[1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', linewidth=0.5, linestyle='--')

    # Annotate Symmetry Points
    for label, (kx, ky) in SYMMETRY_POINTS.items():
        plt.scatter(kx, ky, color='white', s=50, zorder=10)
        # Uncomment to add text labels if desired:
        # plt.text(kx - 0.075, ky - 0.075, label, color='white', fontsize=10)

def plot_spectral_function(
    xx: np.ndarray, 
    yy: np.ndarray, 
    zz: np.ndarray, 
    output_path: Path,
    Save=False
) -> None:
    """
    Generates the main plot, overlays BZ geometry, and saves the file.
    """
    plt.figure(figsize=(6, 5))

    # Plot Color Mesh
    plt.pcolormesh(xx, yy, zz, cmap='gnuplot2', vmin=0.0, vmax=1, shading='auto')
    plt.colorbar(label='Spectral Intensity')

    # Add Contour Lines (Fermi Surface)
    contour = plt.contour(xx, yy, zz, levels=np.linspace(0, 0.8, 9), colors='white', linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

    # Add Symmetry Points
    add_symmetry_points(plt)
    
    # Formatting
    plt.xlim([-0.6, 0.6])
    plt.ylim([-0.6, 0.6])
    plt.xticks([])
    plt.yticks([])
    plt.tick_params(left=False, bottom=False)

    if Save==True:
        # Save and Show
        print(f"Saving plot to: {output_path}")
        plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.show()


def main():
    # Select File
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        return
    print(f"Selected file: {file_path}")

    # Load Data
    try:
        xx_raw, yy_raw, zz_raw = load_and_reshape_data(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Process Data (Symmetry)
    xx, yy, zz = symmetrize_data(xx_raw, yy_raw, zz_raw)

    print(f"Minimum spectral intensity: {np.min(zz):.4f}")
    print(f"Maximum spectral intensity: {np.max(zz):.4f}")

    # Plot
    output_file = file_path.with_suffix('.png')
    plot_spectral_function(xx, yy, zz, output_file)

if __name__ == "__main__":
    main()
