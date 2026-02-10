import numpy as np
from numpy._typing import NDArray

import sys

from math import comb


def binom(n, p):
    k_values = np.arange(0, n+1)
    return np.array([comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in k_values])


def getEverything(Pi, p_0, p_1, eps: float = 0.0001, max_iter: int = 1000):
    M = calculateM(Pi)
    (_, y, _, _) = sinkhornV2(p_0=p_0, p_1=p_1, eps=eps, max_iter=max_iter, M=M)

    return calculatePij(y, Pi)


def calculatePij(y, Pi):
    P = []
    T = len(Pi)
    n = len(y)
    phi = y
    phi_before = np.dot(Pi[T-1], phi)

    for k in range(T):
        t = T-k-1
        p = np.ones((n, n))

        for i in range(n):
            for j in range(n):
                p[i][j] = Pi[t][i][j] * phi[j]/phi_before[i]
        if t-1 >= 0:
            phi = phi_before
            phi_before = np.dot(Pi[t-1], phi)
        P.append(p)
    P.reverse()

    return P


def calculateM(Pi: list[NDArray]):
    T = len(Pi)
    M = Pi[0]

    for i in range(1, T):
        M = np.dot(M, Pi[i])

    return M


def sinkhornV2(p_0: NDArray, p_1: NDArray, eps: float = 0.0001, max_iter: int = 10, M: NDArray | None = None):

    n = p_1.size

    if M is None:
        M = np.ones((n, n))/n

    y = np.ones(n)
    x = np.ones(n)

    cnt = 0

    norm = -1

    while cnt < max_iter:
        xNew = p_0/(np.dot(M, y))

        yNew = p_1/(np.dot(M.T, xNew))

        norm = np.linalg.norm((np.dot(M.T, xNew))*yNew-p_1) + \
            np.linalg.norm((np.dot(M, yNew))*xNew-p_0)
        if norm < eps:
            return (xNew, yNew, norm, cnt)
        x = xNew
        y = yNew
        cnt += 1

    return (x, y, norm, cnt)


def generate_random_permutation_matrix(n: int) -> np.ndarray:
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Dimension 'n' must be a positive integer.")

    permutation = np.random.permutation(n)
    identity_matrix = np.identity(n)
    permutation_matrix = identity_matrix[permutation]

    return permutation_matrix


def generateBinom(n):
    K = n-1

    BinomMat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            BinomMat[i][j] = comb(K, j) * (i/K)**j * (1-i/K)**(K-j)

    return BinomMat


def generate_stochastic_matrix(n):
    matrix = np.random.rand(n, n)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix


def entropy(P, Pi, p_0):
    out = 0
    T = len(Pi)
    n = len(p_0)

    p_actuel = p_0
    for k in range(T):
        startOut = out
        for i in range(n-1):
            partialSum = 0
            for j in range(0, n-1):
                p_ijk = P[k][i][j]
                pi_ijk = Pi[k][i][j]
                if p_ijk != 0 and pi_ijk == 0:
                    return np.inf
                if p_ijk != 0 and pi_ijk != 0:
                    partialSum += p_ijk * np.log(p_ijk/pi_ijk)
            partialSum *= p_actuel[i]
            out += partialSum
        print(out-startOut)
        p_actuel = np.dot(P[k].T, p_actuel)

    return out


if __name__ == "__main__":

    n = int(sys.argv[1])
    max_iter = int(sys.argv[2])
    eps = float(sys.argv[3])

    p_0 = np.zeros(n)
    p_0[0] = 1
    # p_0 = np.ones(n)/n

    p_1 = np.zeros(n)
    p_1[1] = 1/2
    p_1[0] = 1/2

    Unif = np.ones((n, n))/n

    BistochaPrm = generate_random_permutation_matrix(n)

    BinomMat = generateBinom(n)

    RandomMatStochat = generate_stochastic_matrix(n)

    Pi = [RandomMatStochat, generate_stochastic_matrix(n),
          generate_stochastic_matrix(n)

          ]

    P = getEverything(Pi, p_0, p_1, max_iter=max_iter, eps=eps)
    P_test = [Pi[0], Pi[1], P[-1]]
    print(P)
    print(P_test)

    print("algo", entropy(P, Pi, p_0))
    print("test", entropy(P_test, Pi, p_0))
