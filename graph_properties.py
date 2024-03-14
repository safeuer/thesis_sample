#!/usr/bin/python3

from tkinter import W
import networkx as nx
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt
from numpy import linalg as la
import pandas as pd 
from functools import reduce
from math import log


# Graph generation functions
def generate_erdosrenyi(N, prob, n=1):
    ''' Generate erdos renyi
        * Input:
            N: int, number of nodes in graphs
            prob: float, the probability for edge creation in the graphs
            n: int, how many graphs to make
        * Output:
            graphs: list of random Erdos-Renyi networkx graphs
    '''
    graphs = []
    for _ in range(n):
        G = nx.gnp_random_graph(N, prob)
        # Regenerate graph until connected
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(N, prob)
        graphs.append(G)
    return graphs

def generate_barbell(size_complete_graph, length_path):
    """Barbell graphs are deterministic, so this mainly exists for consistency.
    Input: size_complete_graph: int, number of nodes in the complete subgraphs
            length_path: int, number of nodes in the path
    Output: nx.Graph, the barbell graph with the above specifications"""
    return nx.barbell_graph(size_complete_graph, length_path)

def generate_geometric(N, radius, n=1):
    """
    Input: N: number of nodes in graphs
        radius: radius of connectivity in generated graphs
        n: number of graphs to create
    Output: list of random geometric networkx graphs
    """
    graphs = []
    for _ in range(n):
        G = nx.random_geometric_graph(N, radius)
        # Regenerate graph until connected.
        while not nx.is_connected(G):
            G = nx.random_geometric_graph(N, radius)
        G_int = nx.convert_node_labels_to_integers(G)
        graphs.append(G_int)
    return graphs

def generate_grid(dim_X, dim_Y):
    """Grid graphs are deterministic, so this mainly exists for consistency.
    Input: dim_X: int, number of nodes in each geometric row of the graph
            dim_Y: int, number of nodes in each geometric column of the graph
    Output: nx.Graph, grid graph with the above specifications"""
    G = nx.grid_2d_graph(dim_X, dim_Y)
    G_int = nx.convert_node_labels_to_integers(G)
    return G_int   

def generate_regular(N, deg, n=1):
    """
    Input: N: number of nodes in graphs
        deg: degree of each node in graphs
        n: number of graphs to create
    Output: list of CONNECTED random regular networkx graphs
    """
    graphs = []
    for _ in range(n):
        G = nx.random_regular_graph(deg, N)
        # regenerate graphs until connected
        while not nx.is_connected(G):
            G = nx.random_regular_graph(deg, N)
        graphs.append(G)
    return graphs

def generate_cluster_wreath(num_wheels, wheel_size):
    """
    Input: num_wheels: int, number of wheels in the graph
            wheel_size: int, size of each wheel in graph
    Output: nx.Graph, consisting of num_wheels wheel graphs of size wheel_size, where the hub nodes of each wheel are connected.
    """
    # Create num_wheels wheels to connect
    wheels = [nx.wheel_graph(wheel_size) for _ in range(num_wheels)]
    wreath = wheels[0]
    # Connect hub nodes in complete graph
    for i in range(1, num_wheels):
        wreath = nx.disjoint_union(wreath, wheels[i])
        for j in range(i):
            wreath.add_edge(j*wheel_size, i*wheel_size)
    wreath.add_edge(0, (num_wheels-1)*wheel_size)
    return wreath
    


def get_degree_matrix(G):
    """Return diagonal degree matrix of a graph G."""
    N = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    D = np.sum(A, axis=1)
    D = np.array(D).ravel() # Remove extra brackets
    D = np.diag(D) # Make diagonal matrix
    return D

def get_matrices(G):
    ''' D is degree matrix
        A is adjacency matrix
        A_norm is normalized adjacency matrix
        return all 3 of these
    '''
    N = G.number_of_nodes()
    # Adjacency matrix
    A = nx.adjacency_matrix(G)
    A_hat = A.toarray()
    # Degree matrix
    D = get_degree_matrix(G)
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        return None, None, None
    # Alternate normalized adjacency matrix from Sinclair
    #D_half = np.sqrt(D)
    #D_half_inv = np.linalg.inv(D_half)
    #A_norm  = np.matmul( np.matmul(D_half, A_hat), D_half_inv )
    A_norm  = np.matmul( A_hat, D_inv )
    return A, A_norm, D

def get_centrality(G):
    ''' Compute centrality measure for each node in graph G
        * Input:
            G: networkx graph
        * Output:
            PF_vec: vector of centrality values, one for each node
            arg_sorted_PF_vec: 
    '''
    A, A_norm, D = get_matrices(G)
    w, v = np.linalg.eig(A_norm) # Get 
    idx = np.argmax(w)
    vec = v[:,idx] # eigenvector corresponding to max eigenvalue

    # Compute Perron-Frobenius
    PF_vec = vec / sum(vec) # This is the centrality vector
    arg_sorted_PF_vec = np.argsort(PF_vec)
    arg_sorted_PF_vec = arg_sorted_PF_vec[::-1] 
    return PF_vec, arg_sorted_PF_vec

def get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec):
    ''' Input: num_nodes: number of nodes in subgraph (here for convenience)
        nodes: set of nodes in subgraph
        G: full graph
        PF_vec: centrality vector
        arg_sorted_PF_vec: list, order of nodes when sorted by centrality (here for potential improvement)
    '''
    # For nodes already in the set nodes, find their neighbors in G but not in nodes
    neighbors = set()
    for node in nodes:
        for nbr in G.neighbors(node):
            if nbr not in nodes:
                neighbors.add(nbr)
    # Choose neighbor with max centrality and add to set nodes
    max_val = -1
    max_nbr = None
    for nbr in neighbors:
        if PF_vec[nbr] > max_val:
            max_val = PF_vec[nbr]
            max_nbr = nbr
    if max_nbr is not None:
        nodes.add(max_nbr)
    # Get subgraph based on set nodes and their edges
    G0 = G.subgraph(nodes)
    return G0, nodes

def min_dicts(d1, d2):
    """Helper function for shortest_paths_to_subgraph. 
    Input: d1, d2: dicts with identical keys and numerical values.
    Output: dict, with the same keys as d1 and d2 corresponding to the minimum value between them.
    """
    return {k: min(d1[k], d2[k]) for k in d1}


def shortest_paths_to_subgraph(G, subgraph_nodes):
    '''
    Input: G: graph
        subgraph_nodes: list of nodes currently in the subgraph
    Return: dict, with nodes as keys and shortest path length to subgraph as values
    '''
    shortest_paths = []
    for node in subgraph_nodes: # calculate length of shortest path from all nodes to each node in subgraph
        paths = nx.shortest_path_length(G, target=node, weight=None)
        shortest_paths.append(paths)
    min_paths_to_subgraph = reduce(min_dicts, shortest_paths) # Find shortest path among the above dictionaries for each node
    return min_paths_to_subgraph

def one_step_trans_closure(G):
    '''
    Input: G = (V,E), a graph
    Return: a new graph, G_trans = (V,E'), where E' = {E} U {(u,w)|(u,v),(v,w) ∈ E for some v ∈ V}
    '''
    # Transitive closure is A(G) + A(G)^2
    A = nx.to_numpy_array(G, dtype=int)
    A_2 = np.matmul(A, A)
    G_trans = nx.from_numpy_array(A + A_2, parallel_edges=False)
    # reset edge weights
    nx.set_edge_attributes(G_trans, 1, "weight")
    return G_trans

def plot_centralnbr_lambda2(graphs, N, graph_name="", param_value=""):
    '''
    Find central subgraphs from 2 to N nodes for each graph in graphs and plot results.
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of number of nodes vs. mean second eigenvalue
    '''
    df_total = pd.DataFrame(data={"num_nodes": [], "lambda_max2": []})
    for G in graphs:

        # Get centrality vectors
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        # Initialize node set to contain most central node
        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        lambda_max2s = []

        # Build subgraphs with each number of nodes
        for num_nodes in range(2, N):
            #print("-----------------------------")
            #print("num_nodes:", num_nodes)
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            # Get eigenvectors and eigenvalues for subgraph's adjacency matrix
            # w, v = np.linalg.eig(A_norm0 + np.linalg.inv(D0)) # This is what we do if we want to compensate for self-loops.
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)

            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            #print("lambda_max2:", lambda_max2)
            lambda_max2s.append(lambda_max2)
            
        # Create a dataframe to graph (magnitude of) second eigenvalue
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "lambda_max2": lambda_max2s})
        df_total = pd.concat([df_total, df])

    # Calculate intergraph means of second largest eigenvalue magnitude.
    df_means = df_total.groupby(['num_nodes']).mean()

    # Plot number of nodes vs. intergraph mean second eigenvalue.
    # If only 1 graph is specified, no mean is taken, and labels are adjusted accordingly.
    plt.plot(np.arange(start=2, stop=N), df_means['lambda_max2'], 'k-')
    plt.axis([0, N, 0, 1.1])
    plt.xlabel("Subgraph order")
    plt.ylabel("Magnitude of second eigenvalue for subgraph")
    if len(graphs) == 1:
        plt.title(f"Subgraph order vs. magnitude of second eigenvalue for 'closest neighbor' approach on {graph_name}", wrap=True)
    else:
        plt.title(f"Subgraph order vs. magnitude of second eigenvalue for 'closest neighbor' approach on {graph_name} (mean over {len(graphs)} iterations)", wrap=True)
    #plt.savefig(f"./plots/{graph_name}/lambda2/N{nx.number_of_nodes(graphs[0])}/cnbr_L2_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")
    plt.savefig(f"./plots_final/cnbr_L2_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/cnbr_L2_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")

def plot_centralnbr_gap(graphs, N, graph_name="", param_value=""):
    '''
    Find central subgraphs from 2 to N nodes for each graph in graphs and plot results.
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of number of nodes vs. mean second eigenvalue
    '''
    df_total = pd.DataFrame(data={"num_nodes": [], "lambda_max2": []})
    for G in graphs:

        # Get centrality vectors
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        # Initialize node set to contain most central node
        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        spectral_gaps = []

        # Build subgraphs with each number of nodes
        for num_nodes in range(2, N):
            #print("-----------------------------")
            #print("num_nodes:", num_nodes)
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            # Get eigenvectors and eigenvalues for subgraph's adjacency matrix
            # w, v = np.linalg.eig(A_norm0 + np.linalg.inv(D0)) # This is what we do if we want to compensate for self-loops.
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)

            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            #print("lambda_max2:", lambda_max2)
            spectral_gaps.append(1-lambda_max2)
            
        # Create a dataframe to graph (magnitude of) second eigenvalue
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "spectral_gap": spectral_gaps})
        df_total = pd.concat([df_total, df])

    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()

    # Plot number of nodes vs. intergraph mean second eigenvalue.
    # If only 1 graph is specified, no mean is taken, and labels are adjusted accordingly.
    plt.plot(np.arange(start=2, stop=N), df_means['spectral_gap'], 'k-')
    plt.axis([0, N, 0, 1])
    plt.xlabel("Subgraph order")
    plt.ylabel("Spectral gap of subgraph")
    plt.savefig(f"./plots_final/cnbr_gap_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/cnbr_gap_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")

def plot_centralnbr_paths(graphs, N, graph_name="", param_value=""):
    '''
    Plot average shortest path length from nodes to a node in the subgraph, formed using the 'central neighbor' method
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of number of nodes vs. average shortest path to subgraph.
    '''
    df_total = pd.DataFrame(data={"num_nodes": [], "avg_shortest_path": []})
    for G in graphs:
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        # Initialize node set to contain most central node
        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        avg_shortest_paths = []

        for num_nodes in range(2, N):
            #print("-----------------------------")
            #print("num_nodes:", num_nodes)
            #print("nodes0:", nodes)
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            #print(min_paths_to_subgraph)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "avg_shortest_path":avg_shortest_paths})
        df_total = pd.concat([df_total, df])

    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()
    plt.plot(np.arange(start=2, stop=N), df_means['avg_shortest_path'], 'k-')
    plt.axis([0, N, 0, max(df_means['avg_shortest_path'])+1])
    plt.xlabel("Subgraph order")
    plt.ylabel("Mean length of shortest path to subgraph")
    plt.savefig(f"./plots_final/cnbr_dist_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/cnbr_dist_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")   

def plot_centralnbr_cost(graphs, N, alpha=0.5, graph_name="", param_value=""):
    """
    Plot average shortest path length from nodes to a node in the subgraph, formed using the 'central neighbor' method
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    alpha: value of alpha within the cost function.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of number of nodes vs. average shortest path to subgraph.
    """
    df_total = pd.DataFrame(data={"num_nodes": [], "lambda_max2": [], "avg_shortest_path": [], "cost": []})
    for G in graphs:
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        lambda_max2s = []
        avg_shortest_paths = []
        costs = []
        for num_nodes in range(2, N):
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            # Get eigenvectors and eigenvalues for subgraph's adjacency matrix
            # w, v = np.linalg.eig(A_norm0 + np.linalg.inv(D0)) # This is what we do if we want to compensate for self-loops.
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)
            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            lambda_max2s.append(lambda_max2)

            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

            cost = (alpha)*(2*avg_shortest_path) + (1-alpha)*(G.number_of_nodes()/num_nodes)*log(num_nodes)/(1.1-lambda_max2)
            costs.append(cost)
        
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "lambda_max2": lambda_max2s,
                         "avg_shortest_path":avg_shortest_paths, "cost": costs})
        df_total = pd.concat([df_total, df])

    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()

    plt.plot(np.arange(start=2, stop=N), df_means['cost'], 'k-')
    plt.axis([0, N, 0, max(df_means['cost'])+1])
    plt.xlabel("Subgraph order")
    plt.ylabel("Mixing cost")
    plt.savefig(f"./plots_final/cnbr_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/cnbr_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")   

def centralnbr_graphs(graphs, N,  alpha=0.5, graph_name="", param_value=""):
    """
    Plot all 3 of the above graphs.
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    alpha: value of alpha within the cost function.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of each above graph.
    """
    df_total = pd.DataFrame(data={"num_nodes": [], "spectral_gap": [], "avg_shortest_path": [], "cost": []})
    for G in graphs:
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        spectral_gaps = []
        avg_shortest_paths = []
        costs = []
        for num_nodes in range(2, N):
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            # Get eigenvectors and eigenvalues for subgraph's adjacency matrix
            # w, v = np.linalg.eig(A_norm0 + np.linalg.inv(D0)) # This is what we do if we want to compensate for self-loops.
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)
            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            spectral_gaps.append(1-lambda_max2)

            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

            cost = (alpha)*(2*avg_shortest_path) + (1-alpha)*(1/3)*(log(G.number_of_nodes()/num_nodes)+1)*(log(num_nodes)/(1.1-lambda_max2))
            costs.append(cost)
        
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "spectral_gap": spectral_gaps,
                         "avg_shortest_path":avg_shortest_paths, "cost": costs})
        df_total = pd.concat([df_total, df])

    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()

    # Add text to graph in correct location.
    gapmax = np.max(spectral_gaps)
    distmin = np.min(avg_shortest_paths)
    costmin = np.min(costs)

    gapmax_arg = np.argmax(spectral_gaps)
    gapmax_x = np.arange(start=2, stop=N)[gapmax_arg]

    distmin_arg = np.argmin(avg_shortest_paths)
    distmin_x = np.arange(start=2, stop=N)[distmin_arg]

    costmin_arg = np.argmin(costs)
    costmin_x = np.arange(start=2, stop=N)[costmin_arg]

    maxdist = max(df_means['avg_shortest_path'])
    maxcost = max(df_means['cost'])

    if gapmax_x > 3*N/4:
        gap_text_loc = (gapmax_x - (3*N/20), gapmax+0.1)
    else:
        gap_text_loc = (gapmax_x, gapmax+0.1)

    if distmin_x > 3*N/4:
        dist_text_loc = (distmin_x - (3*N/20), distmin-(maxdist/10))
    else:
        dist_text_loc = (distmin_x, distmin-(maxdist/10))
    
    if costmin_x > 3*N/4:
        cost_text_loc = (costmin_x - (3*N/20), costmin-(maxcost/10))
    else:
        cost_text_loc = (costmin_x, costmin-(maxcost/10))

    # Create plots
    _, axes = plt.subplots()
    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['spectral_gap'], 'b-')
    plt.plot(gapmax_x, gapmax, 'r.')
    plt.axis([0, N, 0, 1.1])
    axes.tick_params(axis="both", labelsize=12)
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Spectral gap of subgraph", fontsize=16)
    plt.annotate(f'|S| = {gapmax_x}', xy=(gapmax_x, gapmax), xytext=gap_text_loc, fontsize=14)
    plt.savefig(f"./plots_final/cnbr_gap_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()

    _, axes = plt.subplots()
    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['avg_shortest_path'], 'b-')
    plt.plot(distmin_x, distmin, 'r.')
    plt.axis([0, N, 0, maxdist+1])
    axes.tick_params(axis="both", labelsize=12)
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Mean distance to subgraph", fontsize=16)
    plt.annotate(f'|S| = {distmin_x}', xy=(distmin_x, distmin), xytext=dist_text_loc, fontsize=14)
    plt.savefig(f"./plots_final/cnbr_dist_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()

    _, a = plt.subplots()
    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['cost'], 'b-')
    plt.plot(costmin_x, costmin, 'r.')
    plt.axis([0, N, 0, maxcost+1])
    a.tick_params(axis="both", labelsize=12)
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Mixing cost", fontsize=16)
    plt.annotate(f'|S| = {costmin_x}', xy=(costmin_x, costmin), xytext=cost_text_loc, fontsize=14)
    plt.savefig(f"./plots_final/cnbr_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/cnbr_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")   

def cost_compare(graphs, N, alpha=0.5, graph_name="", param_value=""):
    """
    Legacy function. Compare multiple cost functions to determine normalizing factor between two expressions. 
    """
    df_total = pd.DataFrame(data={"num_nodes": [], "lambda_max2": [], "avg_shortest_path": [], "cost": [], "cost_log":[], "cost_none":[]})
    for G in graphs:
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        lambda_max2s = []
        avg_shortest_paths = []
        costs = []
        cost_logs = []
        cost_nones = []
        for num_nodes in range(2, N):
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            # Get eigenvectors and eigenvalues for subgraph's adjacency matrix
            # w, v = np.linalg.eig(A_norm0 + np.linalg.inv(D0)) # This is what we do if we want to compensate for self-loops.
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)
            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            lambda_max2s.append(lambda_max2)

            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

            cost = (alpha)*(2*avg_shortest_path) + (1-alpha)*(G.number_of_nodes()/num_nodes)*log(num_nodes)/(1.1-lambda_max2)
            costs.append(cost)

            cost_log = (alpha)*(2*avg_shortest_path) + (1-alpha)*(log(G.number_of_nodes()/num_nodes)+1)*log(num_nodes)/(1.1-lambda_max2)
            cost_logs.append(cost_log)

            cost_none = (alpha)*(2*avg_shortest_path) + (1-alpha)*log(num_nodes)/(1.1-lambda_max2)
            cost_nones.append(cost_none)
        
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "lambda_max2": lambda_max2s,
                         "avg_shortest_path":avg_shortest_paths, "cost": costs, "cost_log": cost_logs, "cost_none": cost_nones})
        df_total = pd.concat([df_total, df])

    df_means = df_total.groupby(['num_nodes']).mean()

    alpha_string = str(alpha)[2:] # remove "0."

    plt.plot(np.arange(start=2, stop=N), df_means['cost'], 'b-', label="traffic")
    plt.plot(np.arange(start=2, stop=N), df_means['cost_log'], 'r-', label="log-traffic")
    plt.plot(np.arange(start=2, stop=N), df_means['cost_none'], 'k-', label="none")
    plt.axis([0, N, 0, max(df_means['cost'])//4]) # for clarity
    plt.xlabel("Subgraph order")
    plt.ylabel("Mixing cost")
    if len(graphs) == 1:
        plt.title(f"Subgraph order vs. mixing cost (alpha = {alpha}) for 'central neighbor' approach on {graph_name}", wrap=True)
    else: 
        plt.title(f"Subgraph order vs. mixing cost (alpha = {alpha}) for 'central neighbor' approach on {graph_name} (mean over {len(graphs)} iterations)", wrap=True)
    plt.legend()
    #plt.savefig(f"./plots/{graph_name}/cost/N{nx.number_of_nodes(graphs[0])}/cnbr_cost__N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")
    plt.savefig(f"./plots_final/cost_compare/cnbr_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_A{alpha_string}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()

    df_means.to_csv(f"./csvs_final/cost_compare/cnbr_{graph_name}_N{nx.number_of_nodes(graphs[0])}_A{alpha_string}_P{param_value}_I{len(graphs)}.csv")   

# Creating example graphs
# rand_geo_N100 = nx.random_geometric_graph(100, 0.2, seed=1234)
# rand_geo_N500 = nx.random_geometric_graph(500, 0.2, seed=1234)
# rand_er_N100 = nx.erdos_renyi_graph(100, 0.1, seed=1234)
# rand_er_N500 = nx.erdos_renyi_graph(500, 0.1, seed=1234)
# barb_N100 = nx.barbell_graph(25, 50)
# barb2_N100 = nx.barbell_graph(10, 80)
# barb_N500 = nx.barbell_graph(100, 300)
# barb2_N500 = nx.barbell_graph(25, 450)

if __name__ == "__main__":
    # Example Graphs
    rand_geo_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N100_P02_S1234.txt", nodetype=int), ordering='default')
    rand_geo_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N500_P02_S1234.txt", nodetype=int), ordering='default')
    rand_er_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_er_N100_P01_S1234.txt", nodetype=int), ordering='default')
    rand_er_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_er_N500_P01_S1234.txt", nodetype=int), ordering='default')
    barb_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/barbell_N100_P25.txt", nodetype=int), ordering='default')
    barb2_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/barbell_N100_P10.txt", nodetype=int), ordering='default')
    barb_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/barbell_N500_P100.txt", nodetype=int), ordering='default')
    barb2_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/barbell_N500_P25.txt", nodetype=int), ordering='default')
    rand_reg_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_reg_N100_P5_S1234.txt", nodetype=int), ordering='default')
    rand_reg_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_reg_N500_P10_S1234.txt", nodetype=int), ordering='default')
    wreath_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/wreath_N100_P10.txt", nodetype=int), ordering='default')
    wreath1_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/wreath_N500_P5.txt", nodetype=int), ordering='default')
    wreath2_N500 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/wreath_N500_P100.txt", nodetype=int), ordering='default')

    rand_geo_N250_P01 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N250_P01_S13.txt", nodetype=int), ordering='default')
    rand_geo_N250_P015 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N250_P015_S1234.txt", nodetype=int), ordering='default')
    rand_geo_N250_P02 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N250_P02_S1234.txt", nodetype=int), ordering='default')
    rand_geo_N250_P025 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_geo_N250_P025_S1234.txt", nodetype=int), ordering='default')

    rand_er_N250_P005 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/erdos_renyi_N250_P005_S1234.txt", nodetype=int), ordering='default')
    rand_er_N250_P01 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/erdos_renyi_N250_P01_S1234.txt", nodetype=int), ordering='default')
    rand_er_N250_P015 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/erdos_renyi_N250_P015_S1234.txt", nodetype=int), ordering='default')
    rand_er_N250_P02 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/erdos_renyi_N250_P02_S1234.txt", nodetype=int), ordering='default')

    # # Means

    # rand_geo_N100_P02 = generate_geometric(100, 0.2, 10)
    # rand_er_N100_P01 = generate_erdosrenyi(100, 0.1, 10)
    # rand_reg_N100_P5 = generate_regular(100, 5, n=10)

    # wreath_N256_P4 = generate_cluster_wreath(64, 4)
    # wreath_N256_P8 = generate_cluster_wreath(32, 8)
    # wreath_N256_P16 = generate_cluster_wreath(16, 16)
    # wreath_N256_P32 = generate_cluster_wreath(8, 32)
    # wreath_N256_P64 = generate_cluster_wreath(4, 64)


    # centralnbr_graphs([wreath_N256_P4], 256, graph_name = "wreath", param_value="4")
    # centralnbr_graphs([wreath_N256_P8], 256, graph_name = "wreath", param_value="8")
    # centralnbr_graphs([wreath_N256_P16], 256, graph_name = "wreath", param_value="16")
    # centralnbr_graphs([wreath_N256_P32], 256, graph_name = "wreath", param_value="32")
    # centralnbr_graphs([wreath_N256_P64], 256, graph_name = "wreath", param_value="64")

    # centralnbr_graphs(rand_geo_N100_P02, N=100, graph_name="rand_geo", param_value="02")
    # centralnbr_graphs(rand_er_N100_P01, N=100, graph_name = "erdos_renyi", param_value="01")
    # centralnbr_graphs(rand_reg_N100_P5, N=100, graph_name="rand_reg", param_value="5")

    # # Single Graphs
    # centralnbr_graphs([rand_geo_N100], 100, graph_name="rand_geo", param_value="02")
    # centralnbr_graphs([rand_geo_N500], 500, graph_name="rand_geo", param_value="02")
    # centralnbr_graphs([rand_er_N100], 100, graph_name="erdos_renyi", param_value="01")
    # centralnbr_graphs([rand_er_N500], 500, graph_name="erdos_renyi", param_value="01")
    # centralnbr_graphs([barb_N100], 100, graph_name = "barbell", param_value = "25")
    # centralnbr_graphs([barb_N500], 500, graph_name = "barbell", param_value = "100")
    # centralnbr_graphs([rand_reg_N100], 100, graph_name="rand_reg", param_value="5")
    # centralnbr_graphs([rand_reg_N500], 500, graph_name="rand_reg", param_value="10")
    # centralnbr_graphs([wreath_N100], 100, graph_name="wreath", param_value="10")
    # centralnbr_graphs([wreath1_N500], 500, graph_name="wreath", param_value="5")
    # centralnbr_graphs([wreath2_N500], 500, graph_name="wreath", param_value="100")

    # Parameter compare
    # centralnbr_graphs([rand_geo_N250_P01], 250, graph_name="rand_geo", param_value="01")
    # centralnbr_graphs([rand_geo_N250_P015], 250, graph_name="rand_geo", param_value="015")
    # centralnbr_graphs([rand_geo_N250_P02], 250, graph_name="rand_geo", param_value="02")
    # centralnbr_graphs([rand_geo_N250_P025], 250, graph_name="rand_geo", param_value="025")

    # centralnbr_graphs([rand_er_N250_P005], 250, graph_name="erdos_renyi", param_value="005")
    # centralnbr_graphs([rand_er_N250_P01], 250, graph_name="erdos_renyi", param_value="01")
    # centralnbr_graphs([rand_er_N250_P015], 250, graph_name="erdos_renyi", param_value="015")
    # centralnbr_graphs([rand_er_N250_P02], 250, graph_name="erdos_renyi", param_value="02")

    # for a in [0.1, 0.25, 0.5, 0.75, 0.9]:
    #     cost_compare([rand_geo_N100], 100, alpha=a, graph_name="rand_geo", param_value="02")
    #     cost_compare([rand_geo_N500], 250, alpha=a, graph_name="rand_geo", param_value="02")
    #     cost_compare([rand_er_N100], 100, alpha=a, graph_name="erdos_renyi", param_value="01")
    #     cost_compare([rand_er_N500], 250, alpha=a, graph_name="erdos_renyi", param_value="01")
    #     cost_compare([barb_N100], 100, alpha=a, graph_name = "barbell", param_value = "25")
    #     cost_compare([barb_N500], 250, alpha=a, graph_name = "barbell", param_value = "100")
    #     cost_compare([rand_reg_N100], 100, alpha=a, graph_name="rand_reg", param_value="5")
    #     cost_compare([rand_reg_N500], 250, alpha=a, graph_name="rand_reg", param_value="10")