from typing import Any

from networkx import from_numpy_array, selfloop_edges, k_core, Graph
from numpy import mean, diag, fill_diagonal, array


def adj_to_graph(adj_matrix):
    G = from_numpy_array(adj_matrix)
    return G


def nodes_to_terms(terms, maincore):
    k_core = []
    for i in maincore:
        k_core.append(terms[i])
    return k_core


def calc_average_edge_w(adj_matrix):
    return mean(adj_matrix) / 2


def prune_matrix(adj_matrix, threshold):
    diagonal1 = diag(adj_matrix).copy()
    if threshold > 1:
        adj_matrix[adj_matrix <= threshold] = 0
    fill_diagonal(adj_matrix, diagonal1)
    new_diagonal = adj_matrix.diagonal()
    return adj_matrix



def kcore_nodes(nxgraph, k=None) -> Any:
    nxgraph.remove_edges_from(selfloop_edges(nxgraph))
    try:
        maincore = k_core(nxgraph, k)
    except ValueError:
        maincore = Graph()
        print("k-core decomposition failed")
        print(f"nxgraph: {nxgraph}\n nxgraph.nodes: {nxgraph.nodes}\n nxgraph.edges: {nxgraph.edges}")
        # print(maincore.nodes)
    return maincore.nodes
