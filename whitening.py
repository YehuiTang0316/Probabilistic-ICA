"""
Code for 5 natural whitening algorithms.
"""
import numpy as np
from sklearn.utils import check_array, as_float_array
from sklearn.utils.validation import FLOAT_DTYPES
import warnings

def compute_matrices(data):
    # preprocess data for whitening
    """
    :param data: data waiting for whitening
    :return: cov: covariance matrix
            V: diagonal variance matrix
            P: correlation matrix
            U * Lambda * U.T = cov, U nad Lambda are eigendecomposition of cov
            G * Theta * G.T = P, G and Theta ate eigendecomposition of P
    """
    cov = np.cov(data)
    V = np.diag(np.diag(cov))
    P = np.corrcoef(data)
    Lambda, U = np.linalg.eig(cov)
    Lambda = np.diag(Lambda)
    Theta, G = np.linalg.eig(P)
    Theta = np.diag(Theta)
    return cov, V, P, U, Lambda, G, Theta


class Whitening(object):
    """
    Implement of five methods to do whitening including ZCA-Manhalanobia, ZCA-cor, PCA,
     PCA-cor, Cholesky whitening.

    V: diagonal variance matrix
    P: correlation matrix
    U * Lambda * U.T = cov, U nad Lambda are eigendecomposition of cov
    G * Theta * G.T = P, G and Theta ate eigendecomposition of P

    """
    def __init__(self, data):
        self.data = data
        self.cov, self.V, self.P, self.U, self.Lambda, self.G, self.Theta = compute_matrices(data)

    def ZCA_whitening(self):
        # compute cov ** (-1/2)
        w = np.matmul(np.matmul(self.U, np.sqrt(np.linalg.inv(self.Lambda))), np.transpose(self.U))
        return np.real(np.matmul(w, self.data))

    def PCA_whitening(self):
        # compute Lambda ** (-1/2) * U.T
        w = np.matmul(np.sqrt(np.linalg.inv(self.Lambda)), np.transpose(self.U))
        return np.real(np.matmul(w, self.data))

    def Cholesky_whitening(self):
        # L * L.T = cov ** (-1)
        w = np.transpose(np.linalg.cholesky(np.linalg.inv(self.cov)))
        return np.real(np.matmul(w, self.data))

    def ZCA_cor_whitening(self):
        # compute P ** (-1/2) * V ** (-1/2)
        p = np.matmul(np.matmul(self.G, np.sqrt(np.linalg.inv(self.Theta))), np.transpose(self.G))
        w = np.matmul(p, np.sqrt(np.linalg.inv(self.V)))
        return np.real(np.matmul(w, self.data))

    def PCA_cor_whitening(self):
        # compute Theta ** (-1/2) * G.T * V ** (-1/2)
        w = np.matmul(np.matmul(np.sqrt(np.linalg.inv(self.Theta)), np.transpose(self.G)),
                         np.sqrt(np.linalg.inv(self.V)))
        return np.real(np.matmul(w, self.data))


