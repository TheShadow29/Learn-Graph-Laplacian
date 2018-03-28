import numpy as np
from data_loader import synthetic_data_gen
import pdb


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


def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    #
    M_mat = create_dup_matrix(num_vertices)
    P_mat = 2 * beta * np.dot(M_mat.T, M_mat)
    return M_mat, P_mat


def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec


def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)


def create_dup_matrix(num_vertices):
    M_mat = np.zeros((num_vertices**2, num_vertices*(num_vertices + 1)//2))
    # tmp_mat = np.arange(num_vertices**2).reshape(num_vertices, num_vertices)
    for j in range(1, num_vertices+1):
        for i in range(j, num_vertices+1):
            u_vec = get_u_vec(i, j, num_vertices)
            Tij = get_T_mat(i, j, num_vertices)
            # pdb.set_trace()
            M_mat += np.outer(u_vec, Tij).T

    return M_mat


def create_A_mat(n):
    A_mat = np.zeros((n, n*(n+1)//2))
    A_mat[0, 0] = 1
    A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    # for i in range(1, A_mat.shape[0]+1):
    #     A_mat[i, :] =


if __name__ == "__main__":
    syn = synthetic_data_gen()
    num_nodes = syn.num_vertices
