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

        mean = np.zeros(self.num_vertices)
        cov = np.linalg.pinv(nx.laplacian_matrix(self.er_graph) + np.eye(self.num_vertices) * 0.5)
        # Each row is a signal
        self.graph_signals_random = np.random.multivariate_normal(mean, cov, 100)
        # pdb.set_trace()
        return

# class data_loader:
#     def __init__(self):
