# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Generates colorbar plots
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define the French flag colors

def plotcmap(colourlist):
    """
    Create a color map plot using the given list of colors.

    Args:
        colourlist (list): A list of colors represented as hexadecimal strings.

    Returns:
        None

    Example Usage:
        colourlist = ['#0055a4', '#ffffff', '#ef4135']
        plotcmap(colourlist)

    Code Analysis:
        This function creates a color map plot using the colors in the input list. It uses the LinearSegmentedColormap class from the matplotlib.colors module to create a custom color map. The function then creates a figure and axes using plt.subplots and sets the size of the figure. It generates a gradient array using np.linspace and np.vstack to create a smooth transition of colors. The gradient is then plotted on the axes using ax.imshow with the custom color map. The function also adds text labels to the plot indicating the relevance of the colors. Finally, the plot is saved as an image file.

    """
    cmap = LinearSegmentedColormap.from_list('Custom', colourlist, N=1024)
    fig, ax = plt.subplots(figsize=(25, 1.5))
    gradient = np.linspace(0, 1, 1024)
    gradient = np.vstack((gradient, gradient))

    # Plot the color bar
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    #ax.set_title('Relevance Colour Map',y=1.5,va='center',fontsize=16)
    ax.set_axis_off()

    ax.text(-1, -1, 'negative relevance', ha='left', va='center', color=colourlist[0], fontsize=21)
    ax.text(1024, -1, 'positive relevance', ha='right', va='center', color=colourlist[-1], fontsize=21)
    ax.text(538, -1, 'No relevance', ha='right', va='center', color='black', fontsize=21)
    plt.savefig('./experiment/Notebooks/figures/relevance_colurmap.png', dpi=600, bbox_inches='tight')
    #plt.show()

def plot_hot_colormap(colourlist):
    """
    Create a custom color map plot using a list of colors.

    Args:
        colourlist (list): A list of colors represented as strings.

    Returns:
        None. The function only creates and saves a color map plot.
    """
    # Define the colors for the 'hot' colormap (black to yellow)
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list('CustomHot', colourlist, N=1024)
    fig, ax = plt.subplots(figsize=(25, 1.5))
    gradient = np.linspace(0, 1, 1024)
    gradient = np.vstack((gradient, gradient))

    # Plot the color bar
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    #ax.set_title('Hot Colour Map',y=1.5,va='center',fontsize=16)
    ax.set_axis_off()

    ax.text(-1, -1, 'Not relevant', ha='left', va='center', color=colourlist[0], fontsize=21)
    ax.text(1024, -1, 'Highly relevant', ha='right', va='center', fontsize=21)
    ax.text(768,-1,"moderately relevant",ha='right', va='center', fontsize=21)
    ax.text(265, -1, 'relevant', ha='center', va='center', fontsize=21)
    plt.savefig('./experiment/Notebooks/figures/hot_colormap.png', dpi=600, bbox_inches='tight')
    #plt.show()

hotmapcolourlist = ["black", "red", "yellow", "white"]

plot_hot_colormap(hotmapcolourlist)


colourlist = ['#0055a4', '#ffffff', '#ef4135']
#colourlist=['#FF8000','#ffffff',"#008000"]

plotcmap(colourlist)

