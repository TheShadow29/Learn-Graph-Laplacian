import numpy as np
from data_loader import synthetic_data_gen


def solve_L(Y, alpha, beta):
    L = np.zeros(Y.shape[0], Y.shape[0])
    return L


def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    Returns Output Signal Y, Graph Laplacian L
    """
    y = inp_signal
    ldim = inp_signal.shape[1]
    for _ in range(max_iter):
        # Update L
        L = solve_L(Y, alpha, beta)
        # Update Y
        Y = (np.eye(ldim) + alpha * L)
    return L, Y


if __name__ == "__main__":
    syn = synthetic_data_gen()
