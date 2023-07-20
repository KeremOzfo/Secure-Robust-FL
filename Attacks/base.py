import torch
from client import client
from utils import *
import numpy as np
import math
import torch.nn.functional as F
from math import radians

class _BaseByzantine(client):
    """Base class of Byzantines (omniscient ones).
        Extension of this byzantines are capable of
        gathering data and communication between them.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.adv_momentum = None
        self.psuedo_moments = None
        self.omniscient = True
        self.global_momentum = None

    def omniscient_callback(self,benign_gradients):
        return NotImplementedError

    def get_global_m(self,m):
        self.global_momentum = m
    def adv_pred(self,batch,momentum):
        device = self.device
        x, y = batch
        x, y = x.to(device), y.to(device)
        zero_grad(self.model)
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        self.adv_step(momentum)

    def adv_step(self,momentum):
        args = self.args
        last_ind = 0
        grad_mult = 1 - args.Lmomentum if args.worker_momentum else 1
        for p in self.model.parameters():
            if p.requires_grad:
                d_p = p.grad
                if args.wd != 0:
                    d_p = d_p.add(p.data, alpha=args.wd)
                length, dims = d_p.numel(), d_p.size()
                buf = momentum[last_ind:last_ind + length].view(dims).detach()
                buf.mul_(args.Lmomentum)
                buf.add_(torch.clone(d_p).detach(), alpha=grad_mult)
                momentum[last_ind:last_ind + length] = buf.flatten()  # update buffer

    def train_psuedo_moments(self):
        iterator = iter(self.loader)
        flat_model = get_model_flattened(self.model, self.device)
        if self.psuedo_moments is None:
            b_cl = int(1 / self.args.traitor)
            self.psuedo_moments = [torch.tensor(torch.zeros_like(flat_model)) for cl in range(b_cl)]
        for momentum in self.psuedo_moments:
            momentum.to(self.device)
            for i in range(self.local_steps):
                batch = iterator.__next__()
                self.adv_pred(batch,momentum)
            momentum.to('cpu')
    def get_grad(self):
        return torch.clone(self.adv_momentum).detach()

    def get_benign_preds(self):
        return [torch.clone(m).detach() for m in self.psuedo_moments]



def pert_apprx(pert,std):
    max_pert = torch.where(pert < -std, -std,
                           torch.where(pert >std, std, pert))
    return max_pert

def get_angle(ref,pert):
    angle = math.degrees(math.acos(F.cosine_similarity(pert, ref, dim=0).item()))
    return angle

def bn_mask(net,device):
    import torch.nn as nn
    mask = torch.empty((0),device=device)
    first_conv = True
    for layer in net.modules():
        if isinstance(layer, (nn.BatchNorm2d,nn.Linear)):
            mask_ = torch.ones_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask,mask_),dim=0)
            if layer.bias is not None:
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
        elif isinstance(layer,nn.Conv2d):
            if first_conv:
                mask_ = torch.ones_like(layer.weight.data.detach().flatten())
                first_conv = False
            else:
                mask_ = torch.zeros_like(layer.weight.data.detach().flatten())
            mask = torch.cat((mask, mask_), dim=0)
            if layer.bias is not None: ## if there is no bn
                mask_bias = torch.ones_like(layer.bias.data.detach().flatten())
                mask = torch.cat((mask, mask_bias),dim=0)
    return mask