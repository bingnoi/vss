import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from .transformer.position_encoding import PositionEmbeddingSine
from .transformer.transformer import TransformerEncoder, TransformerEncoderLayer

class TransformerEncoderPixelDecoder(BasePixelDecoder):
    def __init__(self,):
        super.__init__()
        
        
    def forward(self,features):
        for idx,f in enumerate(features):
            x = features[f]