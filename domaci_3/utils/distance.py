import numpy as np


def distance_einsum(Y, sigma_inv, M):
    Z = Y - M
    W = np.einsum("ijk,kl->ijl", Z, sigma_inv)
    d = np.einsum("ijk,ijk->ij", W, Z)
    return d
