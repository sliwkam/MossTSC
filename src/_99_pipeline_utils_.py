"""
This module provides utility functions for visualizing 2D arrays using Matplotlib
and for computing a divider value based on the length of a sequence and an alpha power.
"""

import numpy as np
import matplotlib.pyplot as plt


def visualize_2d_array(arrays, index, title, vmax=1):
    """
    Visualize a 2D array from a list of arrays.

    This function selects a 2D array from the provided list using the specified index,
    displays it as a heatmap with a 'viridis' colormap, and annotates each cell with its value.

    Parameters:
        arrays (list of numpy.ndarray): list of 2D arrays to visualize
        index (int): index of the array in the list to visualize
        title (str): title for the visualization plot
        vmax (float, optional): maximum value for the colormap scale (default is 1)

    Returns:
        None

    Raises:
        IndexError: if the provided index is out of the bounds of the arrays list
    """
    # Ensure index is within the bounds of the array list
    if index < 0 or index >= len(arrays):
        raise IndexError("Index out of bounds for the array list.")
    
    # Select the 2D array at the specified index
    data = arrays[index]
    
    # Create the plot with a specific figure size
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(data, cmap='viridis', vmin=0, vmax=vmax)
    fig.colorbar(cax)
    
    # Annotate the cells with their values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.3f}', va='center', ha='center', color='white', fontsize=8)
    
    # Set the title and display the plot
    ax.set_title(title)
    plt.show()


def get_divider(length, alpha_power):
    """
    Compute a divider based on the given length and alpha power.

    The function performs integer division of the provided length by the alpha power
    to determine an appropriate divider.

    Parameters:
        length (int): the total length of the array or sequence
        alpha_power (int): the alpha power used to compute the divider

    Returns:
        int: the computed divider as an integer
    """

    return int(length // alpha_power)