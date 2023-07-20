from .base import _BaseByzantine
from scipy.stats import norm
import torch
import numpy as np
from utils import count_parameters
from math import sin,cos,radians

class sparse_tmp(_BaseByzantine):
    def __init__(self,n,m,mask,z=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.relocate = True
        self.global_momentum = torch.zeros(count_parameters(self.model), device=self.device)
        self.adv_momentum = torch.zeros(count_parameters(self.model), device=self.device)
        self.sparse_mask = mask
        self.sparse_scale = self.args.sparse_scale
        self.first = True
        s = np.floor(n / 2 + 1) - m
        cdf_value = (n - m - s) / (n - m)
        self.z_alt = norm.ppf(cdf_value)
        self.pi = self.args.pi
        self.lamb = self.args.lamb
        self.psuedo_moments = None
        if z is not None:
            self.z_max = z
        else:
            self.z_max = self.z_alt

    def omniscient_callback(self,benign_gradients):
        # Loop over good workers and accumulate their gradients
        stacked_gradients = torch.stack(benign_gradients, 1)
        mu = torch.mean(stacked_gradients, 1).to(self.device)
        std = torch.std(stacked_gradients, 1).to(self.device)
        ud = self.global_momentum.clone()
        if self.first:
            ud = mu.clone()
            self.first = False
        attack_location = self.global_momentum * self.pi + mu.mul(1 - self.pi)
        lamb = self.args.lamb
        ud = ud.mul(lamb).add(mu, alpha=1 - lamb)  ## reference point
        # if self.args.sparse_cfg == 2: # ALIE + PruneAttack
        #     ##config 2:
        #     final_pert = pert * self.sparse_scale + std * self.z_alt
        # elif self.args.sparse_cfg == 7: # ROP_approx + PruneAttack
        #     final_pert = pert * self.sparse_scale
        #     rop_approxed = rop_hybird(std,ud,self.device,self.args.num_proj)
        #     final_pert.add_(rop_approxed, alpha=self.z_max)
        if self.args.sparse_cfg == 13:  # SuperSparse
            sparse_clp = self.iterative_project(ud,std,self.sparse_mask)
            pert_multer = self.sparse_mask * self.sparse_scale + torch.ones_like(sparse_clp).to(self.device).mul((1-self.sparse_mask) * self.z_max)
            final_pert = sparse_clp * pert_multer
        elif self.args.sparse_cfg == 15:  # Rop_sparsified
            rop_clamp = self.iterative_project(ud, std)
            pert_multer = self.sparse_mask * self.sparse_scale + (1 - self.sparse_mask) * self.z_max
            final_pert = rop_clamp * pert_multer
        elif self.args.sparse_cfg == 16:  # Rop_sparsified
            rop_clamp = self.iterative_project_angled(ud, std)
            pert_multer = self.sparse_mask * self.sparse_scale + (1 - self.sparse_mask) * self.z_max
            final_pert = rop_clamp * pert_multer
        #print(get_angle(ud,final_pert))
        attack = attack_location.add(final_pert)
        self.adv_momentum = attack

    def iterative_project(self, ref, std, sparse_mask=None):
        pert = torch.ones_like(ref).to(self.device)
        if sparse_mask is not None:
            pert.mul_(sparse_mask)
        for i in range(self.args.num_proj):
            pert.sub_(ref.mul((pert @ ref) / (ref @ ref)))
            pert = self.pert_apprx(pert, std)
        pert.mul_(std.norm() / pert.norm())
        return pert
    def iterative_project_angled(self, ref, std, sparse_mask=None):
        pert = torch.ones_like(ref).to(self.device)
        angle = self.args.angle
        sin_, cos_ = sin(radians(angle)), cos(radians(angle))
        ref = ref / ref.norm()
        if sparse_mask is not None:
            pert.mul_(sparse_mask)
        for i in range(self.args.num_proj):
            pert.sub_(ref.mul((pert @ ref) / (ref @ ref)))
            ref = ref / ref.norm()
            pert = pert / pert.norm()
            pert = self.pert_apprx(pert, std)
            pert = (pert * sin_ + ref * cos_)
        pert.mul_(std.norm() / pert.norm())
        return pert

    def pert_apprx(self,pert, std):
        max_pert = torch.where(pert < -std, -std,
                               torch.where(pert > std, std, pert))
        return max_pert