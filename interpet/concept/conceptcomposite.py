from zennit.composites import FlatComposite
# Author: Viswambhar Yasa
# Date: 09-01-2024
# Description: Contains custom composites 
# Contact: yasa.viswambhar@gmail.com
# Additional Comments: 

import torch.nn as nn
from itertools import islice
from zennit.core import Composite
from zennit.canonizers import Canonizer
from zennit.rules import Epsilon,ZPlus,ZBox,AlphaBeta,Pass,Flat
from zennit.types import Linear,Convolution,Activation,AvgPool,BatchNorm
from zennit.composites import SpecialFirstLayerMapComposite,LayerMapComposite,LAYER_MAP_BASE


def module_map(ctx, name, module):
    """
    Maps modules in a neural network to specific rules or operations.

    Args:
        ctx (dict): A dictionary that holds the context information during the mapping process.
        name (str): The name of the module being mapped.
        module (nn.Module): The module being mapped.

    Returns:
        Rule or None: An instance of a rule or None based on the type of the module being mapped.
    """
    try:
        next(module.children())
    except StopIteration:
        pass
    else:
        return None
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return None
    if isinstance(module, nn.Conv2d):
        return AlphaBeta()
    return Epsilon()


class CustomComposite(Composite):
    """
    A class that maps modules in a neural network to specific rules or operations based on a given layer map.
    Built based on the examples found in zennit documents

    Args:
        layer_map (dict): A dictionary mapping layer indices to rules or operations.
        canonizers (list, optional): A list of canonizers. Defaults to None.
    """

    def __init__(self, layer_map, canonizers=None):
        """
        Initializes the CustomComposite instance with a layer map and optional canonizers.

        Args:
            layer_map (dict): A dictionary mapping layer indices to rules or operations.
            canonizers (list, optional): A list of canonizers. Defaults to None.
        """
        self.layers_index = list(layer_map.keys())
        self.layer_length = len(self.layers_index)
        self.layer_map = list(layer_map.values())
        if self.layer_length >= 1:
            super().__init__(module_map=self.layer_mapping, canonizers=canonizers)
        else:
            super().__init__(module_map=self.single_mapping, canonizers=canonizers)

    def layer_mapping(self, ctx, name, module):
        """
        Maps a module to a rule or operation based on the layer map. Handles multi-layer mappings.

        Args:
            ctx (dict): The context dictionary.
            name (str): The name of the module.
            module (nn.Module): The module to be mapped.

        Returns:
            object: The mapped rule or operation, or None if no mapping is found.
        """
        if list(islice(module.children(), 1)):
            return None
        ctx['leafnum'] = ctx.get('leafnum', -1) + 1
        if not isinstance(module, (Linear, Convolution, BatchNorm)):
            return None

        for i in range(self.layer_length - 1):
            if self.layers_index[i] <= ctx['leafnum'] < self.layers_index[i + 1]:
                return self.layer_map[i]
        if ctx['leafnum'] >= self.layers_index[-1]:
            return self.layer_map[-1]
        return None

    def single_mapping(self, ctx, name, module):
        """
        Maps a module to a rule or operation based on the layer map. Handles single-layer mappings.

        Args:
            ctx (dict): The context dictionary.
            name (str): The name of the module.
            module (nn.Module): The module to be mapped.

        Returns:
            object: The mapped rule or operation, or None if no mapping is found.
        """
        if list(islice(module.children(), 1)):
            return None
        ctx['leafnum'] = ctx.get('leafnum', -1) + 1
        if not isinstance(module, (Linear, Convolution, BatchNorm)):
            return None
        else:
            return self.layer_map[0]



class LayerSpecificRuleComposite(Composite):
    """
    A subclass of the `Composite` class in the `zennit.core` module. 
    Maps modules in a neural network to specific rules or operations based on a given layer map.
    build based on the document https://zennit.readthedocs.io
    Args:
        layer_map (dict): A dictionary mapping layer indices to rules or operations.
        canonizers (list, optional): List of canonizers to be applied. Defaults to None.
    """

    def __init__(self, layer_map, canonizers=None):
        """
        Initializes the `LayerSpecificRuleComposite` instance with a layer map and optional canonizers.

        Args:
            layer_map (dict): A dictionary mapping layer indices to rules or operations.
            canonizers (list, optional): List of canonizers to be applied. Defaults to None.
        """
        super().__init__(module_map=layer_map, canonizers=canonizers)
    
    def layer_mapping(self, ctx, name, module):
        """
        Maps a module to a rule or operation based on the layer map. Handles multi-layer mappings.

        Args:
            ctx (dict): The context dictionary.
            name (str): The name of the module.
            module (nn.Module): The module to be mapped.

        Returns:
            object: The rule or operation to be applied to the module.
        """
        # Skip modules with children (non-leaf modules)
        if list(islice(module.children(), 1)):
            return None

        # Increment the leaf module count
        ctx['leafnum'] = ctx.get('leafnum', -1) + 1

        # Apply AlphaBeta rule to Convolution layers
        if isinstance(module, Convolution):
            return ZPlus(alpha=2, beta=1)

        # Apply Epsilon rule to Linear layers
        elif isinstance(module, Linear):
            return ZPlus(epsilon=1e-3)

        # For other types of layers, return None (no specific rule)
        return Pass()

class EpsilonComposite(LayerMapComposite):
    """
    A subclass of LayerMapComposite that creates a composite layer map with epsilon regularization for convolutional and linear layers.
    build based on the document https://zennit.readthedocs.io
    Args:
        epsilon (float, optional): The epsilon value for regularization. Defaults to 1e-5.
        canonizers (list, optional): List of canonizers. Defaults to None.
    """

    def __init__(self, epsilon=1e-5, canonizers=None):
        """
        Initializes the EpsilonComposite object with the specified epsilon value and canonizers.

        Args:
            epsilon (float, optional): The epsilon value for regularization. Defaults to 1e-5.
            canonizers (list, optional): List of canonizers. Defaults to None.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, Epsilon(epsilon=epsilon)),
            (nn.Linear, Epsilon(epsilon=epsilon)),
        ]
        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class ZPlusComposite(LayerMapComposite):
    """
    A composite layer map that applies the ZPlus rule to Convolution and Linear layers.
    build based on the document https://zennit.readthedocs.io
    Args:
        canonizers (list, optional): List of canonizers. Defaults to None.
    """

    def __init__(self, canonizers=None):
        """
        Initializes a ZPlusComposite object.

        Args:
            canonizers (list, optional): List of canonizers. Defaults to None.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (nn.Linear, ZPlus())
        ]
        super().__init__(layer_map=layer_mapping, canonizers=canonizers)

    


class ZPlusEpsilonComposite(LayerMapComposite):
    """
    A subclass of LayerMapComposite that applies the ZPlus rule to Convolution and Linear layers, and the Epsilon rule to Linear layers.
    build based on the document https://zennit.readthedocs.io
    Args:
        epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
        canonizers (list, optional): A list of canonizers to be applied. Defaults to None.
    """
    
    def __init__(self, epsilon=1e-5, canonizers=None):
        """
        Initializes a ZPlusEpsilonComposite object with the specified epsilon value and canonizers.
        
        Args:
            epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
            canonizers (list, optional): A list of canonizers to be applied. Defaults to None.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, ZPlus()),
            (nn.Linear, Epsilon(epsilon=epsilon)),
            (nn.modules.pooling.MaxPool2d, Pass()),
        ]

        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class AlphabetaEpsilonComposite(LayerMapComposite):
    """
    A subclass of LayerMapComposite that applies the AlphaBeta rule to Convolution layers and the Epsilon rule to Linear layers.
     build based on the document https://zennit.readthedocs.io
    Attributes:
    - epsilon (float): The epsilon value for the Epsilon rule. Default is 1e-5.
    - alpha (float): The alpha value for the AlphaBeta rule. Default is 2.
    - beta (float): The beta value for the AlphaBeta rule. Default is 1.
    - canonizers (list): Optional list of canonizers.
    """
    
    def __init__(self, epsilon=1e-5, alpha=2, beta=1, canonizers=None):
        """
        Initializes the AlphabetaEpsilonComposite object with the specified epsilon, alpha, beta values, and optional canonizers.
        
        Parameters:
        - epsilon (float): The epsilon value for the Epsilon rule. Default is 1e-5.
        - alpha (float): The alpha value for the AlphaBeta rule. Default is 2.
        - beta (float): The beta value for the AlphaBeta rule. Default is 1.
        - canonizers (list): Optional list of canonizers.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, AlphaBeta(alpha=alpha, beta=beta)),
            (nn.Linear, Epsilon(epsilon=epsilon)),
        ]

        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class FlatComposite(LayerMapComposite):
    """
    A subclass of LayerMapComposite that applies the Flat rule to certain types of layers in a neural network.
     build based on the document https://zennit.readthedocs.io
    Example Usage:
    ```python

    # Create a FlatComposite object
    composite = FlatComposite()

    # Apply the composite to a neural network
    composite.apply(network)

    # The Flat rule will be applied to Linear, AvgPool, and MaxPool2d layers
    # The Pass rule will be applied to Activation and BatchNorm layers
    ```

    Main functionalities:
    The main functionality of the FlatComposite class is to map specific types of layers in a neural network to the Flat rule. This rule replaces the layer with a single flat rule, which means that the layer is removed and replaced with a single rule that does not modify the input.

    Methods:
    - __init__(self, canonizers=None): Initializes a FlatComposite object with the specified canonizers. If no canonizers are provided, the default canonizers are used.

    Fields:
    - layer_map: A list of tuples that maps specific layer types to their corresponding rules.
    - canonizers: A list of canonizers to be applied to the layers before applying the rules.
    """

    def __init__(self, canonizers=None):
        """
        Initializes a FlatComposite object with the specified canonizers. If no canonizers are provided, the default canonizers are used.

        Args:
        - canonizers (list): A list of canonizers to be applied to the layers before applying the rules. Default is None.
        """
        layer_map = [
            (Linear, Flat()),
            (AvgPool, Flat()),
            (nn.modules.pooling.MaxPool2d, Flat()),
            (Activation, Pass()),
            (BatchNorm, Pass()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


class EpsilonZboxComposite(SpecialFirstLayerMapComposite):
    """
    A subclass of SpecialFirstLayerMapComposite that applies the AlphaBeta rule to Convolution layers and the Epsilon rule to Linear layers. It also applies the ZBox rule to the first Convolution layer.
     build based on the document https://zennit.readthedocs.io
    Args:
        epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
        alpha (float, optional): The alpha value for the AlphaBeta rule. Defaults to 1.
        beta (float, optional): The beta value for the AlphaBeta rule. Defaults to 0.
        low (float, optional): The low value for the ZBox rule. Defaults to -1.
        high (float, optional): The high value for the ZBox rule. Defaults to 1.
        canonizers (list, optional): A list of canonizers to be applied. Defaults to None.
    """

    def __init__(self, epsilon=1e-5, alpha=1, beta=0, low=-1, high=1, canonizers=None):
        """
        Initializes an instance of the EpsilonZboxComposite class with the specified values for epsilon, alpha, beta, low, high, and canonizers.

        Args:
            epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
            alpha (float, optional): The alpha value for the AlphaBeta rule. Defaults to 1.
            beta (float, optional): The beta value for the AlphaBeta rule. Defaults to 0.
            low (float, optional): The low value for the ZBox rule. Defaults to -1.
            high (float, optional): The high value for the ZBox rule. Defaults to 1.
            canonizers (list, optional): A list of canonizers to be applied. Defaults to None.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, AlphaBeta(alpha=alpha, beta=beta)),
            (nn.Linear, Epsilon(epsilon=epsilon)),
        ]
        firstlayer_mapping = [(Convolution, ZBox(low=low, high=high))]
        super().__init__(layer_map=layer_mapping, first_map=firstlayer_mapping, canonizers=canonizers)


class EpsilonFlatComposite(SpecialFirstLayerMapComposite):
    """
    A composite layer map that applies the Epsilon rule to Convolution, Linear, and BatchNorm layers,
    and the Flat rule to the first Convolution layer.
     build based on the document https://zennit.readthedocs.io
    Args:
        epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
        canonizers (list, optional): List of canonizers to apply to the layer maps. Defaults to None.
    """

    def __init__(self, epsilon=1e-5, canonizers=None):
        """
        Initializes an instance of the EpsilonFlatComposite class.

        Args:
            epsilon (float, optional): The epsilon value for the Epsilon rule. Defaults to 1e-5.
            canonizers (list, optional): List of canonizers to apply to the layer maps. Defaults to None.
        """
        layer_mapping = LAYER_MAP_BASE + [
            (Convolution, Epsilon(epsilon=epsilon)),
            (nn.Linear, Epsilon(epsilon=epsilon)),
            (BatchNorm, Epsilon()),
        ]
        firstlayer_mapping = [(Convolution, Flat())]
        super().__init__(layer_map=layer_mapping, first_map=firstlayer_mapping, canonizers=canonizers)
