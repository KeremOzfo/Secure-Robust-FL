import torch
from .base import _BaseAggregator


class CM(_BaseAggregator):
    def __init__(self,*args):
        super(CM,self).__init__()

    def __call__(self, inputs):
        stacked = torch.stack(inputs, dim=0)
        values_upper, _ = stacked.median(dim=0)
        values_lower, _ = (-stacked).median(dim=0)
        return (values_upper - values_lower) / 2
