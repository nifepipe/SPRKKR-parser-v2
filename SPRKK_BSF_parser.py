"""
Imports and processes data from SPRKKR outputs for BSF calculations
Extracs Bloch Spectral Function from .gnu file generated with SPR-KKR and XBAND

Author: David Redka, Nicolas Piwek
Date: 2025.12.09
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tkinter as tk
from tkinter import filedialog
from typing import Tuple
from pathlib import Path

# --- Configuration Constants ---
# Plotting style parameters
TITLE=r"Title"
COLORMAP_STYLE = 'gnuplot2'
MIN_INTENSITY = 0 #arbitrary units
MAX_INTENSITY = 400 #arbitrary units
Y_LIM_MIN = -10 #eV
Y_LIM_MAX = 5 #eV
Y_LABEL_TEXT = '$E - E_F$ (eV)'
LINE_COLOR = 'white'
LINE_STYLE = '-'
LINE_WIDTH = 0.5

# X-axis (k-path) labels for the Band Structure Plot
K_PATH_POSITIONS = [0.0000, 1.0000, 1.8660, 2.5731, 2.9267, 3.9874] #Path length in units of 2pi/a
K_PATH_LABELS = ['X', r'$\Gamma$', 'L', 'W', 'K', r'$\Gamma$'] #Path symmetry point names

# --- Core Functions ---
def select_file() -> Path:
    """
    Opens a file dialog for the user to select a data file.

    Returns:
        Path: The path of the selected file, or an empty string if cancelled.
    """

    root = tk.Tk()
    root.withdraw()
    
    file_filters = [
        ("GNU PlotFiles", "*.gnu"),  # Shows files ending in .gnu
        ("All Files", "*.*")        # Default option to show all files
    ]

    file_path = filedialog.askopenfilename(
        initialdir=".", 
        title="Select the SPR-KKR BSF data file",
        filetypes=file_filters
    )
    
    return Path(file_path)

def load_data(file_path: Path|str) -> np.ndarray:
    """
    Loads numerical data from the file. Expects a .gnu file with no header generated via XBAND.
    
    NOTE: This uses the original np.genfromtxt. This may need to be adjusted as this method is outdated.
    
    Args:
        file_path (Path|str): The path to the data file.

    Returns:
        np.ndarray: A 2D array of numerical data.
    """

    print(f"Loading data from: **{file_path}**")
    data = np.genfromtxt(file_path)
    return data

def reshape_data_to_grid(flat_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reshapes the flat 1D spectral data into 2D arrays (grids) for plotting.
    
    This function handles the potential duplication of k-point (x) values 
    where the k-path wraps around or is repeated in the BSF data format.

    Args:
        flat_data (np.ndarray): A 2D array with (k_path, energy, intensity).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (grid_k_path, grid_energy, grid_intensity)
    """
    # Extract the k-path (x), energy (y), and intensity (z) values.
    k_path_values = flat_data[:, 0]
    energy_values = flat_data[:, 1]
    intensity_values = flat_data[:, 2]

    # Identify unique k-path values and count occurrences.
    unique_k_points, k_point_counts = np.unique(k_path_values, return_counts=True)

    # The standard number of energy-values per k-point (row size of the 2D grid).
    # Assumes the first unique k-point count is the correct number of y-values.
    num_energy_points = k_point_counts[0]
    if not np.all(k_point_counts == num_energy_points):
        print("Warning: K-point counts are not uniform. Adjusting grid size.")
        extra_k_rows = np.sum((k_point_counts - num_energy_points) // num_energy_points)
        total_k_rows = len(k_point_counts) + extra_k_rows
    else:
        total_k_rows = len(unique_k_points)
        
    # The total number of rows in the flat data should match the expected size
    expected_flat_size = total_k_rows * num_energy_points
    if len(k_path_values) != expected_flat_size:
        raise ValueError(
            f"Flat data size ({len(k_path_values)}) does not match "
            f"expected grid size ({total_k_rows} x {num_energy_points} = {expected_flat_size}). "
            "Data reshaping failed. Check the input file format."
        )

    # Reshape the flat arrays into the 2D grid structure.
    grid_k_path = np.reshape(k_path_values, (total_k_rows, num_energy_points))
    grid_energy = np.reshape(energy_values, (total_k_rows, num_energy_points))
    grid_intensity = np.reshape(intensity_values, (total_k_rows, num_energy_points))
    
    print(f"Data reshaped to a grid of shape: {grid_intensity.shape}")
    print(f"Min Intensity: {np.min(grid_intensity):.4f}, Max Intensity: {np.max(grid_intensity):.4f}")
    
    return grid_k_path, grid_energy, grid_intensity

def plot_bsf(
    grid_k_path: np.ndarray, 
    grid_energy: np.ndarray, 
    grid_intensity: np.ndarray, 
    output_file_path: Path|str
) -> None:
    """
    Generates and saves the Band Structure Function plot.

    Args:
        grid_k_path (np.ndarray): The 2D k-path grid.
        grid_energy (np.ndarray): The 2D energy grid.
        grid_intensity (np.ndarray): The 2D intensity grid (Z-data).
        output_file_path (str): Path to save the output image file.
    """
    print("Generating BSF plot...")
    
    plt.figure(figsize=(6, 4))
    
    plt.pcolormesh(
        grid_k_path, 
        grid_energy, 
        grid_intensity, 
        cmap=COLORMAP_STYLE, 
        vmin=MIN_INTENSITY, 
        vmax=MAX_INTENSITY
    )
    
    plt.colorbar(label='Spectral Intensity (a.u.)')
    
    plt.ylim([Y_LIM_MIN, Y_LIM_MAX])
    plt.ylabel(Y_LABEL_TEXT)
    
    k_min=np.min(grid_k_path)
    k_max=np.max(grid_k_path)
    plt.xlim(k_min,k_max)
    scaling_factor=k_max/np.max(K_PATH_POSITIONS)
    new_K_PATH_POSITIONS = [pos*scaling_factor for pos in K_PATH_POSITIONS]
    print(k_max)

    # Set custom x-axis ticks and labels (k-path symmetry points)
    plt.xticks(new_K_PATH_POSITIONS, K_PATH_LABELS)
    plt.xlabel('Wave Vector ($k$)')

    # Add thin vertical lines at symmetry points
    for pos in new_K_PATH_POSITIONS:
        plt.axvline(
            x=pos, 
            color=LINE_COLOR, 
            linestyle=LINE_STYLE, 
            linewidth=LINE_WIDTH
        ) 
    
    # Add a horizontal line at the Fermi energy (E - E_F = 0)
    plt.axhline(
        y=0, 
        color=LINE_COLOR, 
        linestyle=LINE_STYLE, 
        linewidth=LINE_WIDTH
    )
    
    plt.title(TITLE)
    
    # Save the plot
    plt.savefig(output_file_path, dpi=500, bbox_inches='tight')
    print(f"Plot saved to: **{output_file_path}**")
    
    # Show the plot
    plt.show()

def main():
    """
    Main function to orchestrate file selection, data processing, and plotting.
    """

    #File Selection
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting script.")
        sys.exit()
    
    #Data Loading
    try:
        flat_data = load_data(file_path)
    except Exception as e:
        print(f"Error loading data with original function: {e}")
        print("Please consider replacing 'load_data' with an alternative parsing function.")
        sys.exit()

    #Data Reshaping
    try:
        grid_k_path, grid_energy, grid_intensity = reshape_data_to_grid(flat_data)
    except ValueError as e:
        print(f"Error during data reshaping: {e}")
        sys.exit()
        
    #Determine Output Path
    base, _ = os.path.splitext(file_path)
    output_path = base + '_BSF_plot.png' # More descriptive filename

    #Plotting
    plot_bsf(grid_k_path, grid_energy, grid_intensity, output_path)

if __name__ == "__main__":
    main()
