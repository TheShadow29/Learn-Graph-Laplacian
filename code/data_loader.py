import numpy as np
import networkx as nx
import pdb


class synthetic_data_gen:
    def __init__(self, num_vertices=20):
        self.num_vertices = num_vertices
        self.er_prob = 0.2
        self.er_graph = nx.fast_gnp_random_graph(self.num_vertices, self.er_prob)
        # self.num_edges = self.er_graph.number_of_edges()
        # pdb.set_trace()
        self.ba_graph = nx.barabasi_albert_graph(self.num_vertices, 1)
        self.random_graph = nx.random_geometric_graph(self.num_vertices, 0.4)
        for u, v, d in self.random_graph.edges(data=True):
            # pdb.set_trace()
            pos1 = np.array(self.random_graph.node[u]['pos'])
            pos2 = np.array(self.random_graph.node[v]['pos'])
            d['weight'] = np.exp(-np.linalg.norm(pos1 - pos2) / (2 * 0.5 * 0.5))

        self.mean = np.zeros(self.num_vertices)
        self.cov_er = np.linalg.pinv(nx.laplacian_matrix(self.er_graph) +
                                     np.eye(self.num_vertices) * 0.5)
        self.cov_ba = np.linalg.pinv(nx.laplacian_matrix(self.ba_graph) +
                                     np.eye(self.num_vertices) * 0.5)
        self.cov_rand = np.linalg.pinv(nx.laplacian_matrix(self.random_graph) +
                                       np.eye(self.num_vertices) * 0.5)
        self.alpha_rnd = 0.012
        self.beta_rnd = 0.79
        self.thr_rnd = 0.06

        self.alpha_er = 0.0032
        self.beta_er = 0.1
        self.thr_er = 0.08

        self.alpha_ba = 0.0025
        self.beta_ba = 0.05
        self.thr_ba = 0.18
        # pdb.set_trace()
        return

    def get_graph_signals(self):
        # Each row is a signal
        graph_signals_er = np.random.multivariate_normal(self.mean, self.cov_er, 100)
        graph_signals_ba = np.random.multivariate_normal(self.mean, self.cov_ba, 100)
        graph_signals_rand = np.random.multivariate_normal(self.mean, self.cov_rand, 100)
        return (graph_signals_er, graph_signals_ba, graph_signals_rand)
# class data_loader:
#     def __init__(self):x
