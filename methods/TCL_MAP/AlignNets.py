import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

__all__ = ['CTCModule', 'AlignSubNet', 'SimModule']

class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len, args):
        '''
        
        
        '''