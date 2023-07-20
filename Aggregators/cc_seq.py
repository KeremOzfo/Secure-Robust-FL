import math

import numpy as np
import torch
from .base import _BaseAggregator
from numpy.random import default_rng


class Clipping_seq(_BaseAggregator):
    def __init__(self, tau, buck_len=3,buck_avg=False,mult_ref=False):
        self.tau = tau
        self.buck_len = buck_len
        self.buck_avg = buck_avg
        super(Clipping_seq, self).__init__()
        self.momentum = None
        self.mult_ref = mult_ref
        self.cos = torch.nn.CosineSimilarity(dim=0)



    def buck_cos_new(self,inputs):
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = self.buck_len
        cluster_size = len(inputs) // buck_len
        sims = [self.cos(m, self.momentum) for m in inputs]
        sims = torch.tensor(sims, device=device)
        clusters = torch.split(torch.argsort(sims), cluster_size)
        rands = [np.random.choice(cluster_size, len(c), replace=False) for c in clusters]
        buckets = [[] for i in range(cluster_size)]
        for cluster, r in zip(clusters, rands):
            for buck_id, client_id in zip(r, cluster):
                buckets[buck_id].append(inputs[client_id])
        buckets = {i:bucket for i,bucket in enumerate(buckets)}
        return buckets

    def buck_rand_sel(self,inputs):
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        buck_len = 3
        cl_list = np.random.choice(len(inputs), len(inputs), replace=False)
        inputs = torch.stack(inputs)
        num_buck = len(inputs) // buck_len
        cl_list = np.array_split(cl_list, num_buck)
        buckets = []
        for cl_buck in cl_list:
            buckets.append(inputs[cl_buck])
        buckets = {i: bucket for i, bucket in enumerate(buckets)}
        return buckets


    def bucket_cos(self, inputs):
        buck_len = self.buck_len
        n = len(inputs)
        l = math.ceil(n / buck_len)
        cl_list = np.arange(n)
        bucket = {i: [] for i in range(l)}
        cos_sims = [torch.cosine_similarity(self.momentum, i, dim=0).detach().cpu().item() for i in inputs]
        cl_sorted = np.asarray(cos_sims).argsort()
        group_id = [i % l for i in cl_list]
        device = inputs[0].get_device()
        for key in bucket.keys():
            grp = np.asarray(group_id) == key
            new_inputs = [ins.detach().clone().cpu() for ins in inputs]
            new_inputs = np.asarray(new_inputs)
            bucket[key] = new_inputs[cl_sorted[grp]]
        for vals in bucket.values():
            [v.to(device) for v in vals]
        [v.to(device) for v in inputs]
        return bucket

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        flag = 0
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])
            flag = 1
        device = inputs[0].get_device()
        device = device if device > -1 else "cpu"
        if flag:
            bucket = self.buck_cos_new(inputs)
        else:
            bucket = self.buck_rand_sel(inputs)
        orig_ref = self.momentum.detach().clone()
        if self.buck_avg:
            if self.mult_ref:
                for ins in bucket.values():
                    buck_avg = sum(v.to(device) for v in ins) / len(ins)
                    self.momentum = (
                            self.clip(orig_ref + self.clip(buck_avg - orig_ref) - self.momentum)
                            + self.momentum
                    )
            else:
                for val in bucket.values():
                    self.momentum = (
                            self.clip(sum(v.to(device) for v in val) / len(val) - self.momentum)
                            + self.momentum
                    )
        else:
            if self.mult_ref:
                for ins in bucket.values():
                    self.momentum = (
                            sum(self.clip(orig_ref + self.clip(v.to(device) - orig_ref) - self.momentum)
                                for v in ins) / len(ins)
                            + self.momentum
                    )
            else:
                for i, ins in enumerate(bucket.values()):
                    self.momentum = (
                            sum(self.clip(v.to(device) - self.momentum) for v in ins) / len(ins)
                            + self.momentum
                    )
        self.momentum = self.momentum.to(inputs[0])

        return torch.clone(self.momentum).detach()



