# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: This file contains file which are required to plot heatmaps and images of the analysis
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import os
import numpy as np
from PIL import Image
from crp.image import imgify
from zennit.image import imsave
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision.utils import save_image
from typing import Dict, List, Union, Any, Tuple, Iterable


def plot_concepts(heatmapdict, heatmap_path, t_name, figsize=(25, 15)):
    """
    Plot and save a set of heatmaps as images.

    Args:
        heatmapdict (dict): A dictionary containing the heatmaps to be plotted. The keys represent the batch number,
                            and the values are tuples containing a list of channel numbers and the heatmap image.
        heatmap_path (str): The file path to save the heatmap plot as an image.
        t_name (str): The title of the heatmap plot.
        figsize (tuple, optional): The size of the plot figure. Defaults to (25, 15).

    Returns:
        None: The function does not return any value. The heatmap plot is saved as an image file.
    """
    nrows = len(heatmapdict.keys())
    fig, axs = plt.subplots(nrows, 1, figsize=figsize, sharex=False, sharey=True)
    fig.subplots_adjust(hspace=0.1)
    for (batch, img), ax in zip(heatmapdict.items(), axs):
        n = len(img[0])
        width = img[1].width
        height = img[1].height
        sp = np.linspace(width / (2 * n), width - (width / (2 * n)), n)
        ax.imshow(img[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f'image {batch}')
        for i, val in enumerate(img[0]):
            ax.text(sp[i], height + 10, "ch" + str(val), transform=ax.transData, color='black', ha='center',
                    fontsize=16)
    fig.suptitle(t_name, fontsize=16)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    pass

def save_to_image(heatmaps, data, savepath, filename, n_rows, relevance={}, level=2.5):
    """
    Save heatmaps and data as images and combine them into a single image.

    Args:
        heatmaps (numpy array): The heatmaps to be saved as an image.
        data (torch tensor): The data to be saved as an image.
        savepath (str): The directory path where the images will be saved.
        filename (str): The name of the file to be saved.
        n_rows (int): The number of rows in the grid of images.
        relevance (dict, optional): A dictionary of relevance values. Defaults to an empty dictionary.
        level (float, optional): The level of the heatmap. Defaults to 2.5.

    Returns:
        None
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    heatmap_path = os.path.join(savepath, filename + ".png")
    imsave(heatmap_path, heatmaps, cmap="france", level=level, grid=(data.shape[0], n_rows), symmetric=True,
               writer_params={"padding":1})
    image_path = os.path.join(savepath, filename + "_data.png")
    save_image(data, image_path, nrow=1, padding=0)
    combine_images(heatmap_path, image_path, savepath, filename)
    pass


def combine_images(heatmap_path, image_path, savepath, filename):
    """
    Combines a heatmap image and a data image into a single concatenated image.

    Args:
        heatmap_path (str): The file path of the heatmap image.
        image_path (str): The file path of the data image.
        savepath (str): The directory path where the concatenated image will be saved.
        filename (str): The name of the concatenated image file.

    Returns:
        None

    Raises:
        None
    """
    with Image.open(heatmap_path) as heatmap_img, Image.open(image_path) as image_data:
        new_width = heatmap_img.width + image_data.width
        new_height = max(heatmap_img.height, image_data.height)
        # Create a new image with the new dimensions
        comb_image = Image.new('RGB', (new_width, new_height))
        # Paste the two images into the new image
        comb_image.paste(heatmap_img, (0, 0))
        comb_image.paste(image_data, (heatmap_img.width, 0))
        # Save the concatenated image
        combineimage_path = os.path.join(savepath, filename + "_concatenated.png")
        comb_image.save(combineimage_path, format="png")
        pass




def save_grid(ref_c: Dict[int, Any], cmap_dim=1, cmap="bwr", vmin=None, vmax=None, symmetric=True, resize=None, padding=True, level=1, figsize=(6, 6), filepath="."):
    """
    Create a grid of images from a dictionary of image data and save it as an image file. The save_grid is obtained from plot_grid in "zennit-crp" module. 
    We modified the structure to be able to generate better plots and save them with higher quality.

    Args:
        ref_c (Dict[int, Any]): A dictionary containing image data. The keys represent the rows in the grid, and the values are lists of images.
        cmap_dim (int, optional): The dimension along which the colormap is applied. Defaults to 1.
        cmap (str, optional): The colormap to be used. Defaults to "bwr".
        vmin (float, optional): The minimum value of the colormap. Defaults to None.
        vmax (float, optional): The maximum value of the colormap. Defaults to None.
        symmetric (bool, optional): Whether to make the colormap symmetric. Defaults to True.
        resize (int, optional): The size to resize the images. Defaults to None.
        padding (bool, optional): Whether to add padding to the images. Defaults to True.
        level (int, optional): The level of the heatmap. Defaults to 1.
        figsize (Tuple[int, int], optional): The size of the output figure. Defaults to (6, 6).
        filepath (str, optional): The file path to save the grid image. Defaults to ".".

    Raises:
        ValueError: If 'cmap_dim' is not 0 or 1 or None.
        ValueError: If 'ref_c' dictionary does not contain an iterable of torch.Tensor, np.ndarray, PIL Image, or a tuple of thereof.

    Returns:
        None: The function does not return any value. The grid of images is saved as an image file.
    """
    keys = list(ref_c.keys())
    nrows = len(keys)
    value = next(iter(ref_c.values()))

    if cmap_dim > 2 or cmap_dim < 1 or cmap_dim == None:
        raise ValueError("'cmap_dim' must be 0 or 1 or None.")

    if isinstance(value, Tuple) and isinstance(value[0], Iterable):
        nsubrows = len(value)
        ncols = len(value[0])
    elif isinstance(value, Iterable):
        nsubrows = 1
        ncols = len(value)
    else:
        raise ValueError("'ref_c' dictionary must contain an iterable of torch.Tensor, np.ndarray or PIL Image or a tuple of thereof.")

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, 1, wspace=0, hspace=0.2)

    for i in range(nrows):
        inner = gridspec.GridSpecFromSubplotSpec(nsubrows, ncols, subplot_spec=outer[i], wspace=0, hspace=0.1)

        for sr in range(nsubrows):

            if nsubrows > 1:
                img_list = ref_c[keys[i]][sr]
            else:
                img_list = ref_c[keys[i]]
            
            for c in range(ncols):
                ax = plt.Subplot(fig, inner[sr, c])

                if sr == cmap_dim:
                    img = imgify(img_list[c], cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, level=level, resize=resize, padding=padding)
                else:
                    img = imgify(img_list[c], resize=resize, padding=padding)

                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                if sr == 0 and c == 0:
                    ax.set_ylabel(keys[i])

                fig.add_subplot(ax)
                
    outer.tight_layout(fig)  
    fig.savefig(filepath, dpi=300, transparent=True)
    pass