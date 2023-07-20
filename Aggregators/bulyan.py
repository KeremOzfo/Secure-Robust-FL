import torch
from .base import _BaseAggregator

def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)

def multi_krum(distances, n, f):
    """Multi_Krum algorithm
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    m = n-2*f
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]

def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances

def TM(inputs,b):
    if len(inputs) - 2 * b > 0:
        b = b
    else:
        b = b
        while len(inputs) - 2 * b <= 0:
            b -= 1
        if b < 0:
            raise RuntimeError

    stacked = torch.stack(inputs, dim=0)
    largest, _ = torch.topk(stacked, b, 0)
    neg_smallest, _ = torch.topk(-stacked, b, 0)
    new_stacked = torch.cat([stacked, -largest, neg_smallest]).sum(0)
    new_stacked /= len(inputs) - 2 * b
    return new_stacked

class Bulyan(_BaseAggregator):
    def __init__(self, n, m):
        self.n = n
        self.m = m
        super(Bulyan,self).__init__()

    def __call__(self, inputs):
        assert self.n >= 4 * self.m + 3
        distances = pairwise_euclidean_distances(inputs)
        top_m_indices = multi_krum(distances, self.n, self.m)
        values = [inputs[i] for i in top_m_indices]
        aggr_res = TM(values,self.m)
        return aggr_res
