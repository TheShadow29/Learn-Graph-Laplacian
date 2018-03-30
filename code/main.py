import numpy as np
from data_loader import synthetic_data_gen
import pdb
from cvxopt import matrix, solvers
import networkx as nx
from tqdm import tqdm
# def solve_L(Y, alpha, beta):
#     L = np.zeros(Y.shape[0], Y.shape[0])

#     return L


def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    Returns Output Signal Y, Graph Laplacian L
    """
    Y = inp_signal.T
    num_vertices = inp_signal.shape[1]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat = create_static_matrices_for_L_opt(num_vertices, beta)
    # M_c = matrix(M_mat)
    P_c = matrix(P_mat)
    A_c = matrix(A_mat)
    b_c = matrix(b_mat)
    G_c = matrix(G_mat)
    h_c = matrix(h_mat)
    curr_cost = np.linalg.norm(np.ones((num_vertices, num_vertices)), 'fro')
    for it in range(max_iter):
        # pdb.set_trace()
        # Update L
        prev_cost = curr_cost
        q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
        q_c = matrix(q_mat)
        sol = solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        l_vech = np.array(sol['x'])
        l_vec = np.dot(M_mat, l_vech)
        L = l_vec.reshape(num_vertices, num_vertices)
        # Assert L is correctly learnt.
        # assert L.trace() == num_vertices
        assert np.allclose(L.trace(), num_vertices)
        assert np.all(L - np.diag(np.diag(L)) <= 0)
        assert np.allclose(np.dot(L, np.ones(num_vertices)), np.zeros(num_vertices))
        # print('All constraints satisfied')
        # Update Y
        Y = np.dot(np.linalg.inv(np.eye(num_vertices) + alpha * L), inp_signal.T)

        curr_cost = (np.linalg.norm(inp_signal.T - Y, 'fro') +
                     alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                     beta * np.linalg.norm(L, 'fro'))
        # print(curr_cost)
        if np.abs(curr_cost - prev_cost) < 1e-4:
            # print('Stopped at Iteration', it)
            break
        # print
    return L, Y


def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    #
    M_mat = create_dup_matrix(num_vertices)
    P_mat = 2 * beta * np.dot(M_mat.T, M_mat)
    A_mat = create_A_mat(num_vertices)
    b_mat = create_b_mat(num_vertices)
    G_mat = create_G_mat(num_vertices)
    h_mat = np.zeros(G_mat.shape[0])
    return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat


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


def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec


def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    # A_mat[0, 0] = 1
    # A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1

    return A_mat


def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat


def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat


def get_precision_er(w_out, w_gt):
    num_cor = 0
    tot_num = 0
    for r in range(w_out.shape[0]):
        for c in range(w_out.shape[1]):
            if w_out[r, c] > 0:
                tot_num += 1
                if w_gt[r, c] > 0:
                    num_cor += 1
    # print(num_cor, tot_num, num_cor / tot_num)
    return num_cor / tot_num


def get_precision_er_L(L_out, L_gt, thresh=1e-4):
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    # pdb.set_trace()
    W_gt = -L_gt.todense()
    np.fill_diagonal(W_gt, 0)
    return get_precision_er(W_out, W_gt)


def get_recall_er_L(L_out, L_gt, thresh=1e-4):
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    # pdb.set_trace()
    W_gt = -L_gt.todense()
    np.fill_diagonal(W_gt, 0)
    return get_precision_er(W_gt, W_out)


if __name__ == "__main__":
    # np.random.seed(0)
    solvers.options['show_progress'] = False
    syn = synthetic_data_gen()
    num_nodes = syn.num_vertices
    prec_er_list = []
    prec_ba_list = []
    recall_er_list = []
    recall_ba_list = []
    for i in tqdm(range(100)):
        np.random.seed(i)
        graph_signals_er, graph_signals_ba, graph_signals_rand = syn.get_graph_signals()
        L_er, Y_er = gl_sig_model(graph_signals_er, 1000, syn.alpha_er, syn.beta_er)
        L_ba, Y_ba = gl_sig_model(graph_signals_ba, 1000, syn.alpha_er, syn.beta_er)
        L_er_gt = nx.laplacian_matrix(syn.er_graph)
        L_ba_gt = nx.laplacian_matrix(syn.ba_graph)
        prec_er = get_precision_er_L(L_er, L_er_gt, thresh=syn.thr_er)
        prec_ba = get_precision_er_L(L_ba, L_ba_gt, thresh=syn.thr_ba)
        recall_er = get_recall_er_L(L_er, L_er_gt, thresh=syn.thr_er)
        recall_ba = get_recall_er_L(L_ba, L_ba_gt, thresh=syn.thr_ba)

        prec_er_list.append(prec_er)
        recall_er_list.append(recall_er)

        prec_ba_list.append(prec_ba)
        recall_ba_list.append(recall_ba)

    print('Avg Prec ER', np.mean(prec_er_list))
    print('Avg Prec BA', np.mean(prec_ba_list))
    print('Avg Recall ER', np.mean(recall_er_list))
    print('Avg Recall BA', np.mean(recall_ba_list))

    # L_out, Y_out = gl_sig_model(syn.graph_signals_er, 1000, syn.alpha_er, syn.beta_er)
    # # L_out[L_out < 1e-4] = 0
    # W_out = -L_out
    # W_out[W_out < syn.thr_er] = 0
    # np.fill_diagonal(W_out, 0)
    # L_gt = nx.laplacian_matrix(syn.er_graph)
    # W_gt = nx.adjacency_matrix(syn.random_graph)
    # # print('Normed difference', np.linalg.norm(L_out - L_gt))
    # # print('Normed difference', np.linalg.norm(W_out - W_gt))
    # # pdb.set_trace()
    # prec = get_precision_er(W_out, W_gt)
