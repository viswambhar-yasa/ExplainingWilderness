
import torch.nn as nn
from zennit.core import Composite
from itertools import islice
from zennit.composites import SpecialFirstLayerMapComposite,LayerMapComposite,LAYER_MAP_BASE
from zennit.types import Linear,Convolution,Activation,AvgPool,BatchNorm
from zennit.rules import Epsilon,ZPlus,ZBox,AlphaBeta,Pass,Flat


class CustomComposite(Composite):  # Assuming Composite is defined elsewhere
    def __init__(self, layer_map, canonizers=None):
        self.layers_index = list(layer_map.keys())
        self.layer_length = len(self.layers_index)
        self.layer_map = list(layer_map.values())
        if self.layer_length >=1:
            super().__init__(module_map=self.layer_mapping, canonizers=canonizers)
        else:
            super().__init__(module_map=self.single_mapping, canonizers=canonizers)
    
    def layer_mapping(self, ctx, name, module):
        if list(islice(module.children(), 1)):
            return None
        ctx['leafnum'] = ctx.get('leafnum', -1) + 1
        if not isinstance(module, (Linear, Convolution, BatchNorm)):  # Assuming Linear, Convolution, BatchNorm are defined
            return None
        for i in range(self.layer_length - 1):
            if self.layers_index[i] <= ctx['leafnum'] < self.layers_index[i + 1]:
                return self.layer_map[i]
        if ctx['leafnum'] >= self.layers_index[-1]:
            return self.layer_map[-1]
        return None

    def single_mapping(self,ctx,name,module):
        if list(islice(module.children(), 1)):
            return None
        ctx['leafnum'] = ctx.get('leafnum', -1) + 1
        if not isinstance(module, (Linear, Convolution, BatchNorm)):  # Assuming Linear, Convolution, BatchNorm are defined
            return None
        else:
            return self.layer_map[0]


class EpsilonComposite(LayerMapComposite):
    def __init__(self,epsilon=1e-5, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,Epsilon(epsilon=epsilon)),
                       (nn.Linear,Epsilon(epsilon=epsilon)),
                       ]
        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class ZPlusComposite(LayerMapComposite):
    def __init__(self, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,ZPlus()),
                       (nn.Linear,ZPlus()),
                       
                       ]
        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class ZPlusEpsilonComposite(LayerMapComposite):
    def __init__(self,epsilon=1e-5, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,ZPlus()),
                       (nn.Linear,Epsilon(epsilon=epsilon)),
                       (nn.modules.pooling.MaxPool2d, Pass()),
                       ]

        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class AlphabetaEpsilonComposite(LayerMapComposite):
    def __init__(self,epsilon=1e-5,alpha=2,beta=1, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,AlphaBeta(alpha=alpha,beta=beta)),
                       (nn.Linear,Epsilon(epsilon=epsilon)),
                       ]

        super().__init__(layer_map=layer_mapping, canonizers=canonizers)


class FlatComposite(LayerMapComposite):
    def __init__(self, canonizers=None):
        layer_map = [
            (Linear, Flat()),
            (AvgPool, Flat()),
            (nn.modules.pooling.MaxPool2d, Flat()),
            (Activation, Pass()),
            (BatchNorm, Pass()),
        ]
        super().__init__(layer_map, canonizers=canonizers)


class EpsilonZboxComposite(SpecialFirstLayerMapComposite):
    def __init__(self,epsilon=1e-5,alpha=1,beta=0,low=-1,high=1, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,AlphaBeta(alpha=alpha,beta=beta)),
                       (nn.Linear,Epsilon(epsilon=epsilon)),
                       ]
        firstlayer_mapping=[(Convolution,ZBox(low=low,high=high)),
                            ]
        super().__init__(layer_map=layer_mapping, first_map=firstlayer_mapping, canonizers=canonizers)


class EpsilonFlatComposite(SpecialFirstLayerMapComposite):
    def __init__(self,epsilon=1e-5, canonizers=None):
        layer_mapping=LAYER_MAP_BASE+[(Convolution,Epsilon(epsilon=epsilon)),
                       (nn.Linear,Epsilon(epsilon=epsilon)),
                       (BatchNorm, Epsilon()),]
        firstlayer_mapping=[(Convolution,Flat())]
        super().__init__(layer_map=layer_mapping, first_map=firstlayer_mapping, canonizers=canonizers)

