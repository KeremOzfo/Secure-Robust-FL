import numpy as np
import torch

class Clusters(object):
    def __init__(self,args, loyal_clients,traitor_clients):
        self.args = args
        self.benign = loyal_clients
        self.malicous = traitor_clients
        self.clusters = self.get_clusters()

    def shuffle(self):
        self.clusters = self.get_clusters()

    def get_clusters(self):
        num_byz = len(self.benign)
        if self.args.Byz_each_cluster and num_byz>0: #ensures at least 1 byzantine at each cluster
            clusters = np.array_split(self.benign,self.args.num_clusters)
            clusters = [cluster.tolist() for cluster in clusters]
            print(clusters)
            for i in range(num_byz):
                cluster_ind = i % self.args.num_clusters
                clusters[cluster_ind].append(self.malicous[i])
        else: ## full random clustering
            all_clients = [*self.benign, *self.malicous]
            np.random.shuffle(all_clients)
            clusters = np.array_split(all_clients,self.args.num_clusters)
            clusters = [cluster.tolist() for cluster in clusters]
        np.random.shuffle(clusters)
        return clusters

    def get_cluster_grads(self,cluster):
        return [cl.get_grad() for cl in cluster]

    def aggr_clusters(self,aggr=None):
        if aggr is None:
            aggr_clusters = [sum(self.get_cluster_grads(cluster)) / len(cluster) for cluster in self.clusters]
        else: # should not be used with cc
            aggr_clusters = [aggr.__call__(self.get_cluster_grads(cluster)) for cluster in self.clusters]
        return aggr_clusters