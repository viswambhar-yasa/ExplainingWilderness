# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Contains function which perform analysis and plotting the heatmap and other auxilary functions
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 


import math
import torch
import pickle
import numpy as np
from crp.image import imgify
import matplotlib.pyplot as plt

from sentinelmodels.pretrained_models import buildmodel
from interpret.concept.conceptrelevance import ConceptRelevance


def save_dict(dictionary, filename):
    """
    Save a dictionary to a file using pickle serialization.

    Args:
        dictionary (dict): The dictionary object to be saved.
        filename (str): The name of the file to save the dictionary.

    Returns:
        None

    Example:
        my_dict = {'name': 'John', 'age': 30}
        save_dict(my_dict, 'data.pkl')

    """
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)

def load_dict(filename):
    """
    Load a dictionary object from a file using pickle serialization.

    Args:
        filename (str): The name of the file to load the dictionary from.

    Returns:
        dict: The dictionary object loaded from the file.
    """
    with open(filename, 'rb') as f:
        dictionary = pickle.load(f)
    return dictionary

def compute_lrp(model_type, n_classes, modelweightpath, images, condition=[{"y": 1}], compositname="epsilonplus", canonizerstype="vgg", outputtype="max", device="cuda"):
    """
    Compute the relevance heatmaps for a given set of images using the LRP (Layer-wise Relevance Propagation) technique.

    Args:
        model_type (str): The type of model architecture to use.
        n_classes (int): The number of classes for the classification task.
        modelweightpath (str): The path to the model weights.
        images (torch.Tensor): The input images for which the relevance heatmaps will be computed.
        condition (list, optional): The conditions for relevance analysis. Defaults to [{"y": 1}].
        compositname (str, optional): The name of the composite function to use for relevance propagation. Defaults to "epsilonplus".
        canonizerstype (str, optional): The type of canonizers to use for relevance propagation. Defaults to "vgg".
        outputtype (str, optional): The type of output to consider for relevance analysis. Defaults to "max".
        device (str, optional): The device on which the analysis will be performed. Defaults to "cuda".

    Returns:
        tuple: A tuple containing the relevance heatmaps and relevance values.

    Example:
        heatmap, relevance = compute_lrp(model_type='alexnet', n_classes=2, modelweightpath='path/to/weights', images=images, condition=[{"y": 1}], compositname="epsilonplus", canonizerstype="vgg", outputtype="max", device="cuda")
    """
    model = buildmodel(model_type=model_type, multiclass_channels=n_classes, modelweightpath=modelweightpath).to(device)
    Concepts = ConceptRelevance(model, device=device)
    if condition is None:
        if isinstance(n_classes, int):
            condition = [{"y": i} for i in range(n_classes)]
    recordlayers = list(Concepts.layer_map.keys())
    heatmap, relevance, _ = Concepts.conditional_relevance(images, condition, compositname, canonizerstype, outputtype, recordlayers)
    return heatmap, relevance

def plot_and_annotate_max_min_index(original_arr, label, color, marker, reverse=True,maxposition=(0,10),minposition=(0,-15),fontsize=16):
    """
    Plots a scatter plot of an input array, highlighting the maximum and minimum values.
    Annotates the plot with the corresponding indices of the maximum and minimum values.

    Args:
        original_arr (list): The input array of values.
        label (str): The label for the scatter plot.
        color (str): The color of the scatter plot markers.
        marker (str): The marker style for the scatter plot markers.
        reverse (bool, optional): Whether to sort the input array in reverse order. Defaults to True.
        maxposition (tuple, optional): The offset position for the annotation of the maximum value. Defaults to (0, 10).
        minposition (tuple, optional): The offset position for the annotation of the minimum value. Defaults to (0, -15).
        fontsize (int, optional): The font size for the annotations. Defaults to 16.

    Returns:
        None
    """
    sorted_arr = sorted(original_arr, reverse=reverse)
    max_value_index = np.argmax(original_arr)  # Index of max value in the original array
    min_value_index = np.argmin(original_arr)  # Index of min value in the original array
    max_value_sorted_index = sorted_arr.index(original_arr[max_value_index])  # Find the index in the sorted array
    min_value_sorted_index = sorted_arr.index(original_arr[min_value_index])  # Find the index in the sorted array

    plt.scatter(np.arange(len(original_arr)), sorted_arr, label=label, color=color, marker=marker)
    plt.scatter(max_value_sorted_index, sorted_arr[max_value_sorted_index], color=color, s=100, edgecolors='black',marker="d")  # Highlight the max index in the sorted array
    plt.scatter(min_value_sorted_index, sorted_arr[min_value_sorted_index], color=color, s=100, edgecolors='black',marker="d")  # Highlight the min index in the sorted array
    
    # Adjust text position based on the index's location
    max_text_offset = maxposition 
    min_text_offset = minposition 

    plt.annotate(f"Max Ch: {max_value_index}", (max_value_sorted_index, sorted_arr[max_value_sorted_index]), textcoords="offset points", xytext=max_text_offset,color=color, ha='center',fontsize=fontsize)
    plt.annotate(f"Min Ch: {min_value_index}", (min_value_sorted_index, sorted_arr[min_value_sorted_index]), textcoords="offset points", xytext=min_text_offset,color=color, ha='center',fontsize=fontsize)

def get_info(dictionary):
    """
    Returns a dictionary containing information about the shape, mean relevance, and channel mean of the values in the input dictionary.

    Args:
        dictionary (dict): A dictionary containing tensors as values.

    Returns:
        dict: A dictionary containing information about the shape, mean relevance, and channel mean of the values in the input dictionary.
    """
    rel_featuresdic={}
    for i,value in dictionary.items():
        dimshape = (-1, -2) if len(value.shape) > 2 else -1
        layerdict={"layershape":value.shape,
       "meanrel":value.mean(dim=0).sum().to("cpu").tolist()
        ,"channelmean":value.mean(dim=0).sum(dim=dimshape).to("cpu").tolist()}
        rel_featuresdic[i]=layerdict
    return rel_featuresdic

def create_bar_chart(data_dict, filepath="./bar_plot.png", fig_width=None, figheight=2.5):
    """
    Generates a bar chart with annotations and lines to visualize the data in the input dictionary.

    Args:
        data_dict (dict): A dictionary containing the data to be visualized. The keys are the names of the layers, and the values are the corresponding values for each layer.
        filepath (str, optional): The path to save the bar chart image. Defaults to "./bar_plot.png".
        fig_width (float, optional): The width of the figure in inches. Defaults to None.
        figheight (float, optional): The height of the figure in inches. Defaults to 2.5.

    Returns:
        None
    """
    uniform_height = 2
    bar_width = 0.4
    layers = list(data_dict.keys())
    values = list(data_dict.values())

    # Check if the last value is "output" to set outputline
    outputline = (values[-1] == "output")
    if outputline:
        values.pop()  # Remove the last value "output"
    if fig_width is None:
        fig_width = 3 + len(layers) * 0.85  # Adjust the multiplier as needed
    plt.figure(figsize=(fig_width, figheight))

    # Creating bars with uniform height
    bars = plt.bar(layers, [uniform_height] * len(layers), color='#ACD0F4', width=bar_width, edgecolor='grey')
    for bar in bars:
        bar.set_linestyle('--')  # Set the line style to dashed
        bar.set_linewidth(2.0)  # Set the line width to 2.0

    # Adding layer names at the center of the bars
    for bar, layer in zip(bars, layers):
        plt.text(bar.get_x() + bar.get_width() / 2, uniform_height / 2, layer,
                 ha='center', va='center', color='black', fontsize=14, rotation=90)
        linecolour = "blue"
        if data_dict[layer] < 0:
            linecolour = "red"
        plt.vlines(bar.get_x() + bar.get_width() / 2, uniform_height, uniform_height + 0.25, color=linecolour,
                   linestyles='dashed', linewidth=1.5)
        plt.text(bar.get_x() + bar.get_width() / 2, uniform_height + 0.25,
                 str(round((100 - abs(data_dict[layer])), 1)) + "%", ha='center', va='bottom', color='black',
                 fontsize=14)

    # Adding values at the top of each bar
    for value, layer in zip(values, layers):
        textcolor = "red"
        if value < 0:
            textcolor = "blue"
        plt.text(layer, -0.3, str(round(abs(value), 1)) + "%",
                 ha='center', va='bottom', color=textcolor, fontsize=14)

    for i in range(len(layers) - 1):
        weight = abs(values[i])  # Weight for the current layer
        linewidth = 2 * round((weight / 100), 1) + 1
        if values[i + 1] > 0:
            plt.hlines(uniform_height / 2, bars[i].get_x() + bar_width, bars[i + 1].get_x(), color='red',
                       linestyle='solid', linewidth=linewidth)
        else:
            plt.hlines(uniform_height / 2, bars[i].get_x() + bar_width, bars[i + 1].get_x(), color='blue',
                       linestyle='solid', linewidth=linewidth)
    textcolor = "red"
    if values[0] < 0:
        textcolor = "blue"
    plt.hlines(uniform_height / 2, bars[0].get_x(), bars[0].get_x() - bar_width, color=textcolor, linestyle='solid',
               linewidth=2 * round(abs(values[0] / 100), 1) + 1)
    if outputline:
        plt.hlines(uniform_height / 2, bars[-1].get_x() + bar_width, bars[-1].get_x() + 2 * bar_width, color='black',
                   linestyle='solid', linewidth=3)

    plt.xticks([])  # Hiding x-axis ticks as layer names are inside the bars
    plt.yticks([])  # Adjusting y-limit for better visibility of top labels

    plt.tight_layout()
    plt.axis("off")
    # Show the plot
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=600)

    pass
    

def generate_layer_heatmap(relevance_dict, cmap="hot", symmetric=False, grid=None, level=1.2):
    """
    Generate feature maps for each layer in the given relevance dictionary.

    Args:
        relevance_dict (dict): A dictionary containing the relevance maps for each layer.
            The keys are the layer names and the values are the relevance maps.
        cmap (str, optional): The colormap to use for visualizing the feature maps.
            Defaults to "hot".
        symmetric (bool, optional): Whether to make the colormap symmetric.
            Defaults to False.
        grid (tuple, optional): The grid size for arranging the feature maps.
            Defaults to None.
        level (float, optional): The level of the colormap.
            Defaults to 1.2.

    Returns:
        dict: A dictionary containing the generated feature maps for each layer.
            The keys are the layer names and the values are the corresponding feature maps.
    """
    relevance_feature_map = {}
    for layer_map, relevance in relevance_dict.items():
        if len(relevance.shape) > 2:
            relevance_map = torch.sum(relevance, dim=1)
            relevance_feature_map[layer_map] = imgify(relevance_map.to("cpu").numpy(), cmap=cmap, symmetric=symmetric, grid=(relevance.shape[0], 1), level=level)
    return relevance_feature_map



def visualize_layerheatmap(relevance_feature_map, recordlayers=None, title="Mean Relevance Across Convolution Layers", filepath="./layerheatmap.png", rows=None, columns=None, fontsize=25, figsize=None):
    """
    Create a grid of subplots displaying heatmaps of the relevance feature map for specific layers.

    Args:
        relevance_feature_map (dict): A dictionary containing the relevance feature maps for different layers.
            The keys are the layer names, and the values are the corresponding relevance feature maps.
        recordlayers (list, optional): A list of layer names to be visualized. If not provided, all layers in the relevance_feature_map will be visualized.
        title (str, optional): The title of the plot. Defaults to "Mean Relevance Across Convolution Layers".
        filepath (str, optional): The path to save the plot image. Defaults to "./layerheatmap.png".
        rows (int, optional): The number of rows in the grid layout. Defaults to None.
        columns (int, optional): The number of columns in the grid layout. Defaults to None.
        fontsize (int, optional): The font size for the plot title and layer names. Defaults to 25.
        figsize (tuple, optional): The size of the figure in inches. Defaults to None.

    Returns:
        None. The function generates a plot and saves it as an image file.
    """
    if recordlayers is None:
        recordlayers = list(relevance_feature_map.keys())
    num_images = len(recordlayers)
    if columns is None:
        columns = int(math.ceil(math.sqrt(num_images)))
    if rows is None:
        rows = int(math.ceil(num_images / columns))
    if figsize is None:
        figsize = (10 * columns, 15 * rows)
    fig, axes = plt.subplots(rows, columns, figsize=(figsize))
    fig.suptitle(title, fontsize=fontsize)
    if rows > 1 or columns > 1:
        axes = axes.flatten()

    # Loop through the dictionary and add each image to a subplot
    for i, layername in enumerate(recordlayers):
        if rows == 1 and columns == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(relevance_feature_map[layername])  # Display the image
        ax.set_title(layername, fontsize=fontsize)  # Set the subplot title
        ax.axis('on')  # Turn on the axis
        ax.get_yaxis().set_visible(False)  # Hide the y-axis
        # Turn off the axis

    # Turn off any extra subplots
    for j in range(i + 1, rows * columns):
        axes[j].axis('off')
    plt.tight_layout()  # Adjust subplots to fit into the figure area
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=600)


def remove_duplicates(lst):
    """
    Remove duplicates from a list.

    Args:
        lst (list): The input list containing elements.

    Returns:
        list: A new list containing only the unique elements from the input list.

    Example:
        >>> lst = [1, 2, 3, 2, 4, 3, 5]
        >>> unique_list = remove_duplicates(lst)
        >>> print(unique_list)
        [1, 2, 3, 4, 5]
    """
    seen = set()
    unique_list = []
    for item in lst:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list


def generate_condition_list(subconceptdict, prediction=[1]):
    """
    Generate a list of conditions based on the values in subconceptdict.

    Args:
        subconceptdict (dict): A dictionary containing layer names as keys and lists of channel indices as values.
        prediction (list, optional): A list representing the prediction. Default value is [1].

    Returns:
        list: A list of dictionaries representing different conditions based on the values in subconceptdict. Each dictionary contains the keys 'y', 'layer1', 'layer2', and so on, with their corresponding values.
    """
    conditionlist=[]
    basedict={"y":prediction}
    layernameslist=list(subconceptdict.keys())
    firstlayername=layernameslist[-1]
    for key,value in subconceptdict.items():
        subconceptdict[key]=remove_duplicates(value)
    for channelindex in subconceptdict[firstlayername]:
        basedict[firstlayername]=[channelindex]
        conditionlist.append(basedict)

    for layername in layernameslist[:-1]:
        inputlayername,outputlayername=layername[1].split(":")
        for channelindex in subconceptdict[layername]:
            inputchannel,outputchannel=channelindex[1].split(":")
            for condition in conditionlist:
                if inputlayername in condition:
                    if condition[inputlayername]==[int(inputchannel)]:
                        if outputlayername not in condition:
                            condition[outputlayername]=[int(outputchannel)]
                        elif not condition[outputlayername]==[int(outputchannel)]:
                            newcondition=dict(condition)
                            newcondition[outputlayername]=[int(outputchannel)]
                            conditionlist.append(newcondition)
                            break
    return conditionlist