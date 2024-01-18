import os
import copy
import time
import pickle
import numpy as np
import scipy as sp
import torch

from options import args_parser
from logging_results import logging

from opacus import PrivacyEngine
import opacus
from opacus.accountants.analysis import rdp as privacy_analysis

from tqdm import tqdm

MNIST_SIZE = 60000
CIFAR10_SIZE = 50000

if __name__ == '__main__':
    args = args_parser()  
    print(args)
    
    if args.dataset_name=='mnist':
        dataset_size = MNIST_SIZE
    elif args.dataset_name=='cifar10':
        dataset_size = CIFAR10_SIZE
    
    size_per_user = dataset_size/args.num_client
    total_iterations = int(1e4) # needs discussion
        
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    if args.noise_from_cluster:
        client_per_cluster = int(args.num_client/args.num_clusters) # can we arrange them so that the division is an integer?
        effective_std = args.sigma*args.bs*client_per_cluster/args.clip_val
    else:
        effective_std = args.sigma*args.bs/args.clip_val

    
    rdp = privacy_analysis.compute_rdp(
                    q= 0.1, #args.local_bs/dataset_size, # we need to determine this
                    noise_multiplier=effective_std,
                    steps=total_iterations,
                    orders=alphas,
                )
    
    eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=args.delta
        )
    
    print(f"epsilon:{eps}, delta:{args.delta}")
    
