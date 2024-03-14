import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log
from functools import reduce
from scipy.sparse import csr_matrix

from graph_properties import get_subgraph, get_centrality, get_matrices, shortest_paths_to_subgraph, sort_nodes, generate_erdosrenyi, generate_geometric, generate_regular, generate_cluster_wreath

def add_supernode(G, subgraph):
    """
    Input: G, nx.Graph
        subgraph: nx.Graph, a subgraph of G
    Returns: G', new_node
        G': nx.Graph, G with one additional node that is connected to all nodes in subgraph
        new_node: the name of said new node
    """
    G2 = G.copy()
    new_node = G2.order() + 1 # Name of new node is first unused integer.
    G2.add_node(new_node)
    # Create edge from new to each node in subgraph.
    G2.add_edges_from([(new_node, x) for x in subgraph.nodes()]) 
    return G2, new_node

def add_trans_closure_edges(G, subgraph, num_steps):
    """
    Input: G, nx.Graph
        subgraph: nx.Graph, a subgraph of G
        num_steps, int, the number of steps to take in the transitive closure
    Returns: nx.Graph, which contains all edges in the num_steps-step transitive closure of the subgraph.
    """
    G2 = G.copy()
    #
    A = nx.adjacency_matrix(subgraph)
    A_hat = A.copy()
    adj_matrices_to_sum = [A_hat]
    for _ in range(2):
        A_hat = A_hat.multiply(A)
        adj_matrices_to_sum.append(A_hat)
    trans_matrix = reduce(np.add, adj_matrices_to_sum, csr_matrix(np.zeros(np.shape(A))))
    subgraph_trans = nx.from_numpy_array(trans_matrix, parallel_edges=False)
    # following will add 'duplicate' edges, but that's OK
    G2.add_edges_from(subgraph_trans.edges)
    nx.set_edge_attributes(G2, 1, "weight")
    return G2

def super_improvement(graphs, N, alpha=0.5, graph_name="", param_value=""):
    """
    Plot the number of nodes vs. spectral gap and mixing cost for the original and supergraphs.
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    alpha: value of alpha within the cost function.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of each above graph.
    """
    df_total = pd.DataFrame(data={"num_nodes": [], "spectral_gap": [], "avg_shortest_path": [], "cost": [], 
        "super_spectral_gap":[], "super_cost": []})
    for G in graphs:
        # Initiate subgraph as most central node.
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        spectral_gaps = []
        avg_shortest_paths = []
        costs = []
        super_spectral_gaps = []
        super_costs = []
        # Perform MCN algorithm to create sequence of subgraphs.
        for num_nodes in range(2, N):
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)

            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)
            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            spectral_gaps.append(1-lambda_max2)

            # Calculate travel distance to subgraph and cost.
            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

            cost = (alpha)*(2*avg_shortest_path) + (1-alpha)*(1/3)*(log(G.number_of_nodes()/num_nodes)+1)*(log(num_nodes)/(1.1-lambda_max2))
            costs.append(cost)

             # Calculate spectral gap and cost function for super-subgraph.
            G_supered, new_node = add_supernode(G, G0)
            G_supered0 = G_supered.subgraph(nodes | {new_node})

            At0, At_norm0, Dt0 = get_matrices(G_supered0)
            wt, vt = np.linalg.eig(At_norm0) 
            idxt = np.argmax(wt)

            wt_prime = np.delete(wt, idxt)
            idxt = np.argmax(np.abs(wt_prime))
            lambda_max2t = np.abs(wt_prime[idxt])
            super_spectral_gaps.append(1-lambda_max2t)

            costT = (alpha)*(2*avg_shortest_path) + (1-alpha)*(1/3)*(log(G.number_of_nodes()/(num_nodes+1))+1)*(log(num_nodes+1)/(1.1-lambda_max2t))
            super_costs.append(costT)
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "spectral_gap": spectral_gaps,
                "avg_shortest_path":avg_shortest_paths, "cost": costs, "super_spectral_gap": super_spectral_gaps, "super_cost": super_costs})
        df_total = pd.concat([df_total, df])
    
    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()

    # Determine placement of text.
    gapmax = np.max(spectral_gaps)
    costmin = np.min(costs)
    super_gapmax = np.max(super_spectral_gaps)
    super_costmin = np.min(super_costs)

    gapmax_arg = np.argmax(spectral_gaps)
    gapmax_x = np.arange(start=2, stop=N)[gapmax_arg]

    costmin_arg = np.argmin(costs)
    costmin_x = np.arange(start=2, stop=N)[costmin_arg]

    super_gapmax_arg = np.argmax(super_spectral_gaps)
    super_gapmax_x = np.arange(start=2, stop=N)[super_gapmax_arg]

    super_costmin_arg = np.argmin(super_costs)
    super_costmin_x = np.arange(start=2, stop=N)[super_costmin_arg]


    maxcost = max(df_means['cost'])

    if gapmax_x > 3*N/4:
        gap_text_loc = (gapmax_x - (3*N/20), gapmax+0.02)
    else:
        gap_text_loc = (gapmax_x, gapmax+0.02)
    
    if costmin_x > 3*N/4:
        cost_text_loc = (costmin_x - (3*N/20), costmin-(maxcost/20))
    else:
        cost_text_loc = (costmin_x, costmin-(maxcost/20))

    if super_gapmax_x > 3*N/4:
        super_gap_text_loc = (super_gapmax_x - (3*N/20), super_gapmax+0.02)
    else:
        super_gap_text_loc = (super_gapmax_x, super_gapmax+0.02)
    
    if super_costmin_x > 3*N/4:
        super_cost_text_loc = (super_costmin_x - (3*N/20), super_costmin-(maxcost/20))
    else:
        super_cost_text_loc = (super_costmin_x, super_costmin-(maxcost/20))

    # Create plots.
    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['spectral_gap'], 'b-', label="original")
    plt.plot(np.arange(start=2, stop=N), df_means['super_spectral_gap'], 'r--', label="with super node")
    plt.plot(gapmax_x, gapmax, color="cornflowerblue", marker=".")
    plt.plot(super_gapmax_x, super_gapmax, color="lightcoral", marker=".")
    plt.axis([0, N, 0, 1.1])
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Spectral gap of subgraph", fontsize=16)
    plt.annotate(f'({gapmax_x}, {round(gapmax, 2)})', xy=(gapmax_x, gapmax), xytext=gap_text_loc, color="navy", fontsize=14)
    plt.annotate(f'({super_gapmax_x}, {round(super_gapmax, 2)})', xy=(super_gapmax_x, super_gapmax), xytext=super_gap_text_loc, color="maroon", fontsize=14)
    plt.legend()
    plt.savefig(f"./plots_final/super/super_gap_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()

    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['cost'], 'b-', label="original")
    plt.plot(np.arange(start=2, stop=N), df_means['super_cost'], 'r--', label="with super node")
    plt.plot(costmin_x, costmin, color="cornflowerblue", marker=".")
    plt.plot(super_costmin_x, super_costmin, color="lightcoral", marker=".")
    plt.axis([0, N, 0, max(df_means['cost'])+1])
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Mixing cost", fontsize=16)
    plt.annotate(f'({costmin_x}, {round(costmin, 2)})', xy=(costmin_x, costmin), xytext=cost_text_loc, color="navy", fontsize=14)
    plt.annotate(f'({super_costmin_x}, {round(super_costmin, 2)})', xy=(super_costmin_x, super_costmin), xytext=super_cost_text_loc, color="maroon", fontsize=14)
    plt.legend()
    plt.savefig(f"./plots_final/super/super_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/super/super_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")   

def trans_improvement(graphs, N, num_steps=1, alpha=0.5, graph_name="", param_value=""):
    """
    Plot the number of nodes vs. spectral gap and mixing cost for the original and trans-subgraphs.
    graphs: List of graphs to use for calculation.
    N: maximum number of nodes to create subgraph for.
    num_steps: number of steps of transitive closure to take.
    alpha: value of alpha within the cost function.
    graph_name: name for graph used. Only for file naming purposes.
    param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, saves of a plot and csv of each above graph.
    """
    df_total = pd.DataFrame(data={"num_nodes": [], "spectral_gap": [], "avg_shortest_path": [], "cost": [], 
        "trans_spectral_gap":[], "trans_cost": []})
    for G in graphs:
        PF_vec, arg_sorted_PF_vec = get_centrality(G)

        nodes = set()
        nodes.add(arg_sorted_PF_vec[0])

        spectral_gaps = []
        avg_shortest_paths = []
        costs = []
        trans_spectral_gaps = []
        trans_costs = []
        for num_nodes in range(2, N):
            G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
            A0, A_norm0, D0 = get_matrices(G0)
            w, v = np.linalg.eig(A_norm0) 
            idx = np.argmax(w)
            # Remove largest eigenvalue and find next largest eigenvalue. 
            w_prime = np.delete(w, idx)
            idx = np.argmax(np.abs(w_prime))
            lambda_max2 = np.abs(w_prime[idx])
            spectral_gaps.append(1-lambda_max2)

            # Calculate travel distance to subgraph and mixing cost.
            min_paths_to_subgraph = shortest_paths_to_subgraph(G, nodes)
            nonzero_path_lengths = [v for v in min_paths_to_subgraph.values() if v > 0]
            avg_shortest_path = np.mean(nonzero_path_lengths)
            avg_shortest_paths.append(avg_shortest_path)

            cost = (alpha)*(2*avg_shortest_path) + (1-alpha)*(1/3)*(log(G.number_of_nodes()/num_nodes)+1)*(log(num_nodes)/(1.1-lambda_max2))
            costs.append(cost)

            # Calculate spectral gap and mixing cost for trans-subgraph. 
            G_transed = add_trans_closure_edges(G, G0, num_steps)
            G_transed0 = G_transed.subgraph(nodes)

            At0, At_norm0, Dt0 = get_matrices(G_transed0)
            wt, vt = np.linalg.eig(At_norm0) 
            idxt = np.argmax(wt)

            wt_prime = np.delete(wt, idxt)
            idxt = np.argmax(np.abs(wt_prime))
            lambda_max2t = np.abs(wt_prime[idxt])
            trans_spectral_gaps.append(1-lambda_max2t)

            costT = (alpha)*(2*avg_shortest_path) + (1-alpha)*(1/3)*(log(G.number_of_nodes()/num_nodes)+1)*(log(num_nodes)/(1.1-lambda_max2t))

            trans_costs.append(costT)
        df = pd.DataFrame(data={"num_nodes": np.arange(start=2, stop=N), "spectral_gap": spectral_gaps,
                "avg_shortest_path":avg_shortest_paths, "cost": costs, "trans_spectral_gap": trans_spectral_gaps, "trans_cost": trans_costs})
        df_total = pd.concat([df_total, df])
    
    # Take mean of each statistic if multiple graphs are given as inputs.
    df_means = df_total.groupby(['num_nodes']).mean()

    # Determine text placement.
    gapmax = np.max(spectral_gaps)
    costmin = np.min(costs)
    trans_gapmax = np.max(trans_spectral_gaps)
    trans_costmin = np.min(trans_costs)

    gapmax_arg = np.argmax(spectral_gaps)
    gapmax_x = np.arange(start=2, stop=N)[gapmax_arg]

    costmin_arg = np.argmin(costs)
    costmin_x = np.arange(start=2, stop=N)[costmin_arg]

    trans_gapmax_arg = np.argmax(trans_spectral_gaps)
    trans_gapmax_x = np.arange(start=2, stop=N)[trans_gapmax_arg]

    trans_costmin_arg = np.argmin(trans_costs)
    trans_costmin_x = np.arange(start=2, stop=N)[trans_costmin_arg]


    maxcost = max(df_means['cost'])

    if gapmax_x > 3*N/4:
        gap_text_loc = (gapmax_x - (3*N/20), gapmax+0.02)
    else:
        gap_text_loc = (gapmax_x, gapmax+0.02)
    
    if costmin_x > 3*N/4:
        cost_text_loc = (costmin_x - (3*N/20), costmin-(maxcost/20))
    else:
        cost_text_loc = (costmin_x, costmin-(maxcost/20))

    if trans_gapmax_x > 3*N/4:
        trans_gap_text_loc = (trans_gapmax_x - (3*N/20), trans_gapmax+0.02)
    else:
        trans_gap_text_loc = (trans_gapmax_x, trans_gapmax+0.02)
    
    if trans_costmin_x > 3*N/4:
        trans_cost_text_loc = (trans_costmin_x - (3*N/20), trans_costmin-(maxcost/20))
    else:
        trans_cost_text_loc = (trans_costmin_x, trans_costmin-(maxcost/20))

    # Create plots.
    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['spectral_gap'], 'b-', label="original")
    plt.plot(np.arange(start=2, stop=N), df_means['trans_spectral_gap'], 'g-.', label=f"{num_steps}-step trans. closure")
    plt.plot(gapmax_x, gapmax, color="cornflowerblue", marker=".")
    plt.plot(trans_gapmax_x, trans_gapmax, color="yellowgreen", marker=".")
    plt.axis([0, N, 0, 1.1])
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Spectral gap of subgraph", fontsize=16)
    plt.annotate(f'({gapmax_x}, {round(gapmax, 2)})', xy=(gapmax_x, gapmax), xytext=gap_text_loc, color="navy", fontsize=14)
    plt.annotate(f'({trans_gapmax_x}, {round(trans_gapmax, 2)})', xy=(trans_gapmax_x, trans_gapmax), xytext=trans_gap_text_loc, color="darkgreen", fontsize=14)
    plt.legend()
    plt.savefig(f"./plots_final/trans/trans_gap_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()

    plt.figure(figsize = (7, 5), dpi=500)
    plt.plot(np.arange(start=2, stop=N), df_means['cost'], 'b-', label="original")
    plt.plot(np.arange(start=2, stop=N), df_means['trans_cost'], 'g-.', label=f'{num_steps}-step trans. closure')
    plt.plot(costmin_x, costmin, color="cornflowerblue", marker=".")
    plt.plot(trans_costmin_x, trans_costmin, color="yellowgreen", marker=".")
    plt.axis([0, N, 0, max(df_means['cost'])+1])
    plt.xlabel("Subgraph order", fontsize=16)
    plt.ylabel("Mixing cost", fontsize=16)
    plt.annotate(f'({costmin_x}, {round(costmin, 2)})', xy=(costmin_x, costmin), xytext=cost_text_loc, color="navy", fontsize=14)
    plt.annotate(f'({trans_costmin_x}, {round(trans_costmin, 2)})', xy=(trans_costmin_x, trans_costmin), xytext=trans_cost_text_loc, color="darkgreen", fontsize=14)
    plt.legend()
    plt.savefig(f"./plots_final/trans/trans_cost_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.png")    
    plt.clf()
    df_means.to_csv(f"./csvs_final/trans/trans_{graph_name}_N{nx.number_of_nodes(graphs[0])}_P{param_value}_I{len(graphs)}.csv")   

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

    # Compare Parameters
    # super_improvement([rand_geo_N250_P01], 250, graph_name="rand_geo", param_value="01")
    # super_improvement([rand_geo_N250_P015], 250, graph_name="rand_geo", param_value="015")
    # super_improvement([rand_geo_N250_P02], 250, graph_name="rand_geo", param_value="02")
    # super_improvement([rand_geo_N250_P025], 250, graph_name="rand_geo", param_value="025")

    # super_improvement([rand_er_N250_P005], 250, graph_name="erdos_renyi", param_value="005")
    # super_improvement([rand_er_N250_P01], 250, graph_name="erdos_renyi", param_value="01")
    # super_improvement([rand_er_N250_P015], 250, graph_name="erdos_renyi", param_value="015")
    # super_improvement([rand_er_N250_P02], 250, graph_name="erdos_renyi", param_value="02")

    # trans_improvement([rand_geo_N250_P01], 250, graph_name="rand_geo", param_value="01")
    # trans_improvement([rand_geo_N250_P015], 250, graph_name="rand_geo", param_value="015")
    # trans_improvement([rand_geo_N250_P02], 250, graph_name="rand_geo", param_value="02")
    # trans_improvement([rand_geo_N250_P025], 250, graph_name="rand_geo", param_value="025")

    # trans_improvement([rand_er_N250_P005], 250, graph_name="erdos_renyi", param_value="005")
    # trans_improvement([rand_er_N250_P01], 250, graph_name="erdos_renyi", param_value="01")
    # trans_improvement([rand_er_N250_P015], 250, graph_name="erdos_renyi", param_value="015")
    # trans_improvement([rand_er_N250_P02], 250, graph_name="erdos_renyi", param_value="02")

    # Wreath params

    # wreath_N256_P4 = generate_cluster_wreath(64, 4)
    # wreath_N256_P8 = generate_cluster_wreath(32, 8)
    # wreath_N256_P16 = generate_cluster_wreath(16, 16)
    # wreath_N256_P32 = generate_cluster_wreath(8, 32)
    # wreath_N256_P64 = generate_cluster_wreath(4, 64)

    #super_improvement([wreath_N256_P4], 256, graph_name="wreath", param_value="4")
    # super_improvement([wreath_N256_P8], 256, graph_name="wreath", param_value="8")
    # super_improvement([wreath_N256_P16], 256, graph_name="wreath", param_value="16")
    # super_improvement([wreath_N256_P32], 256, graph_name="wreath", param_value="32")
    # super_improvement([wreath_N256_P64], 256, graph_name="wreath", param_value="64")

    #trans_improvement([wreath_N256_P4], 256, graph_name="wreath", param_value="4")
    # trans_improvement([wreath_N256_P8], 256, graph_name="wreath", param_value="8")
    # trans_improvement([wreath_N256_P16], 256, graph_name="wreath", param_value="16")
    # trans_improvement([wreath_N256_P32], 256, graph_name="wreath", param_value="32")
    # trans_improvement([wreath_N256_P64], 256, graph_name="wreath", param_value="64")

    # # Single Graphs
    # super_improvement([rand_geo_N100], 100, graph_name="rand_geo", param_value="02")
    # super_improvement([rand_geo_N500], 500, graph_name="rand_geo", param_value="02")
    # super_improvement([rand_er_N100], 100, graph_name="erdos_renyi", param_value="01")
    # super_improvement([rand_er_N500], 500, graph_name="erdos_renyi", param_value="01")
    # super_improvement([barb_N100], 100, graph_name = "barbell", param_value = "25")
    # super_improvement([barb_N500], 500, graph_name = "barbell", param_value = "100")
    # super_improvement([rand_reg_N100], 100, graph_name="rand_reg", param_value="5")
    # super_improvement([rand_reg_N500], 500, graph_name="rand_reg", param_value="10")
    # super_improvement([wreath_N100], 100, graph_name="wreath", param_value="10")
    # super_improvement([wreath1_N500], 500, graph_name="wreath", param_value="5")
    # super_improvement([wreath2_N500], 500, graph_name="wreath", param_value="100")

    # trans_improvement([rand_geo_N100], 100, graph_name="rand_geo", param_value="02")
    # trans_improvement([rand_geo_N500], 500, graph_name="rand_geo", param_value="02")
    # trans_improvement([rand_er_N100], 100, graph_name="erdos_renyi", param_value="01")
    # trans_improvement([rand_er_N500], 500, graph_name="erdos_renyi", param_value="01")
    # trans_improvement([barb_N100], 100, graph_name = "barbell", param_value = "25")
    # trans_improvement([barb_N500], 500, graph_name = "barbell", param_value = "100")
    # trans_improvement([rand_reg_N100], 100, graph_name="rand_reg", param_value="5")
    # trans_improvement([rand_reg_N500], 500, graph_name="rand_reg", param_value="10")
    # trans_improvement([wreath_N100], 100, graph_name="wreath", param_value="10")
    # trans_improvement([wreath1_N500], 500, graph_name="wreath", param_value="5")
    # trans_improvement([wreath2_N500], 500, graph_name="wreath", param_value="100")