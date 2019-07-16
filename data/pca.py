import numpy as np


def pca(k, data_handler, center=True):
    X,y = data_handler.get_vectorized()

    if center:
        X = X - np.mean(X)

    # Calculates the Scatter matrix
    S = np.dot(X, X.T)

    # Computes the eigenvalues and its vectors
    W, v = np.linalg.eigh(S)

    # Sort the eigenvalues
    sorted_W = np.argsort(W)

    # Gets the k eigenvectors asscociated with the largest k eigenvalues
    kW = np.array([v[:, i] for i in np.flip(sorted_W[-k:])]).T

    return np.dot(kW.T, X), kW


def rbf_pca(k, data_handler, sigma=0.1, center=True):
    K = __get_gaussian_kernel_matrix(data_handler, sigma)

    if center:
        n = K.shape[0]
        ones = np.ones(n)
        f = (np.identity(n) - np.dot(ones, ones.T) / n)
        K = np.dot(f, np.dot(K, f))

    # Computes the eigenvalues and its vectors from the centered matrix
    A, v = np.linalg.eigh(K)

    # Sort the eigenvalues
    sorted_A = np.argsort(A)

    # Gets the k eigenvectors asscociated with the largest k eigenvalues
    kA = np.array([v[:, i] for i in np.flip(sorted_A[-k:])]).T

    return np.dot(kA.T, K), kA


def __get_gaussian_kernel_matrix(data_handler, sigma=0.1):
    X, y = data_handler.get_vectorized()
    X = np.transpose(X)

    return np.array([[np.exp(-(1/(2*sigma**2))*np.linalg.norm(X[i] - X[j])**2) for j in range(X.shape[0])] for i in range(X.shape[0])])
