from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np
from utils import count_parameters
import math
from math import radians

class reloc(_BaseByzantine): ## ortho to ps & proj start rand
    def __init__(self, n, m, z=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relocate = True
        self.first = True
        self.global_momentum = torch.zeros(count_parameters(self.model), device=self.device)
        if z is not None:
            self.z_max = z
        else:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = norm.ppf(cdf_value)
        self.mask = torch.ones_like(self.global_momentum)
        self.pi = self.args.pi

    def omniscient_callback_old(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1)
        ud = self.global_momentum.clone()
        lamb = self.args.lamb
        if self.first:
            ud = mu.clone()
            self.first = False
        ud = ud.mul(lamb).add(mu,alpha=1-lamb)
        pert = torch.ones_like(mu).to(self.device)
        proj_pert = ud.mul((pert@ud)/(ud@ud))
        pert.sub_(proj_pert)
        z = self.z_max / pert.norm()
        attack = self.global_momentum.add(pert.mul(z))
        self.adv_momentum = attack

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1)
        #std = torch.std(stacked_gradients, 1)
        attack_location = self.global_momentum * self.pi + mu.mul(1-self.pi)
        ud = self.global_momentum.clone()
        lamb = self.args.lamb
        if self.first:
            ud = mu.clone()
            self.first = False
        ud = ud.mul(lamb).add(mu,alpha=1-lamb) ## reference point
        pert = torch.ones_like(mu).to(self.device)
        proj_pert = ud.mul((pert @ ud) / (ud @ ud))
        pert.sub_(proj_pert)
        n_ud = ud / ud.norm()
        pert = pert / pert.norm()
        angle = self.args.angle
        sin, cos = math.sin(radians(angle)), math.cos(radians(angle))
        pert = (pert * sin + n_ud * cos)
        z = self.z_max / pert.norm()
        attack = attack_location.add(pert.mul(z))
        self.adv_momentum = attack