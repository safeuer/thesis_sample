import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from graph_properties import get_subgraph, get_centrality, generate_cluster_wreath, generate_geometric
from math import sqrt
from expansion_improvement import add_trans_closure_edges, add_supernode
from random import uniform

def draw_topology(G, pos_attr=None, node_color='r', graph_name="", param_value=""):
    """
    Draw the graph with the given settings.
    Input: G: nx.Graph to plot
            pos_attr: str ('pos' if the node attribute 'pos' contains node positions; 'circle' if a circle layout is desired; and anything else if a spring layout is desired)
            node_color: str, node color
            graph_name: name for graph used. Only for file naming purposes.
            param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Output: none, saves plot of G with given settings
    """
    fig, ax = plt.subplots()
    plt.figure(figsize = (5, 5), dpi=500)
    # Determine node positions
    if pos_attr == "pos": 
        pos = nx.get_node_attributes(G, pos_attr)
    elif pos_attr == "circle":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k = 2/sqrt(G.number_of_nodes()), seed=1234)
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_color, alpha=0.8, linewidths=0.5)
    nx.draw_networkx_edges(G, pos, width=0.75)
    plt.savefig(f"./graphics/topo_{graph_name}_N{G.number_of_nodes()}_P{param_value}.png")
    #plt.show()
    plt.clf()

def draw_barbell(G, M1, M2, node_color='r', graph_name="", param_value=""):
    """
    Draw the BARBELL graph with the given settings.
    Input: G: nx.Graph to plot (barbell)
            M1: int, size of complete subgraphs in G
            M2: int, length of path in G
            node_color: str, desired node color
            graph_name: name for graph used. Only for file naming purposes.
            param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Output: none, saves plot of G with given settings
    """
    fig, ax = plt.subplots()
    N = G.number_of_nodes()
    plt.figure(figsize = (5, 5), dpi=500)
    # Create dictionary of node positions. First subgraph near top of window randomly distributed in square.
    pos = {k: (2.5+uniform(-1, 1), 3.5+uniform(-0.5,0.5)) for k in range(0, M1)}
    # Second subgraph near bottom of window randomly distributed in square.
    pos.update({N-k-1: (2.5+uniform(-1, 1), 1.5+uniform(-0.5, 0.5)) for k in range(0, M1)})
    # Path connecting two subgraphs given positions that minimize confusion.
    pos.update({M1+k: (2.5+uniform(-0.25, 0.25), 3-(k/M2)) for k in range(0, M2)})
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_color, alpha=0.8, linewidths=0.5)
    nx.draw_networkx_edges(G, pos, width=0.75)
    plt.savefig(f"./graphics/topo_{graph_name}_N{G.number_of_nodes()}_P{param_value}.png")
    #plt.show()
    plt.clf()

def draw_subgraph_choices(G, ns, pos_attr=None, subgraph_col='navy',non_subgraph_col='lightskyblue', M1=None,M2=None, graph_name="", param_value=""):
    """
    Draw the MCN subgraph choice for each order N_s in ns highlighted within G.
    Input: G: nx.Graph to plot
            ns: list of ints, each MCN subgraph order to plot
            pos_attr: str ('pos' if the node attribute 'pos' contains node positions; 
            'circle' if a circle layout is desired; 'barbell' if a barbell layout is desired; and anything else if a spring layout is desired)
            subgraph_col: str, color for nodes in the subgraph
            non_subgraph_col: str, color for nodes not in the subgraph
            M1: int, size of complete subgraphs in G (only for barbell graphs)
            M2: int, length of path in G (only for barbell graphs)
            graph_name: name for graph used. Only for file naming purposes.
            param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, plots the MCN subgraph choices for each N_s in ns, highlighted within G
    """
    # We only need to run the MCN algorithm until we have considered all possible N_s in ns
    N = max(ns)
    PF_vec, arg_sorted_PF_vec = get_centrality(G)

    nodes=set()
    nodes.add(arg_sorted_PF_vec[0])

    # Keep track of MCN subgraphs using node attributes
    for num_nodes in range(2, N+1):
        G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
        if num_nodes in ns: # Mark nodes in subgraph with 1 in their node attribute to indicate they are in that subgraph order.
            nx.set_node_attributes(G, 0, f'subgraph{num_nodes}')
            for node in nodes:
                G.nodes[node][f'subgraph{num_nodes}'] = 1
    
    # Create one drawing for each subgraph order
    fig, axs = plt.subplots(nrows=1, ncols=len(ns), figsize=(4*len(ns), 4), dpi=500)
    # Choose node positioning
    if pos_attr == "pos": 
        pos = nx.get_node_attributes(G, pos_attr)
    elif pos_attr == "circle":
        pos = nx.circular_layout(G)
    elif pos_attr == "barbell":
        pos = {k: (2+uniform(-1, 1), 3+uniform(-0.4,0.4)) for k in range(0, M1)}
        pos.update({G.number_of_nodes()-k-1: (2+uniform(-1, 1), 1+uniform(-0.4, 0.4)) for k in range(0, M1)})
        pos.update({M1+k: (2+uniform(-0.25, 0.25), 2.5-(k/M2)) for k in range(0, M2)})
    else:
        pos = nx.spring_layout(G, k = 2/sqrt(G.number_of_nodes()), seed=1234)
    for i in range(len(ns)):
        # Use node attributes to find list of nodes within subgraph
        subgraph_indicators = nx.get_node_attributes(G, f'subgraph{ns[i]}')
        subgraph_nodes = [node for (node,ind) in subgraph_indicators.items() if ind == 1]
        non_subgraph_nodes = [node for (node,ind) in subgraph_indicators.items() if ind == 0]
        if len(ns) == 1:
            # Plot nodes within subgraph and then nodes outside of subgraph
            nx.draw_networkx_nodes(G, pos, nodelist=subgraph_nodes, node_color=subgraph_col, node_size=50, alpha=1, linewidths=0.5)
            nx.draw_networkx_nodes(G, pos, nodelist=non_subgraph_nodes, node_color=non_subgraph_col, node_size=50, alpha=0.8, linewidths=0.5) 
            nx.draw_networkx_edges(G, pos, width=0.4)
        else:
            nx.draw_networkx_nodes(G, pos, nodelist=subgraph_nodes, ax=axs[i], node_color=subgraph_col, node_size=50, alpha=1, linewidths=0.5)
            nx.draw_networkx_nodes(G, pos, nodelist=non_subgraph_nodes, ax=axs[i], node_color=non_subgraph_col, node_size=50, alpha=0.8, linewidths=0.5) 
            nx.draw_networkx_edges(G, pos, ax=axs[i], width=0.4)
    plt.savefig(f"./graphics/subgraphs_{graph_name}_N{G.number_of_nodes()}_P{param_value}_ns{ns[0]}.png")   
    plt.clf()

def draw_trans_subgraph(G, N, pos_attr=None, subgraph_col='navy',non_subgraph_col='lightskyblue', graph_name="", param_value=""):
    """
    Draw the enhanced MCN trans-subgraph of order N highlighted within G, alongside the unenhanced subgraph.
    Input: G: nx.Graph to plot
            N: subgraph order to plot
            pos_attr: str ('pos' if the node attribute 'pos' contains node positions; 'circle' if a circle layout is desired; and anything else if a spring layout is desired)
            subgraph_col: str, color for nodes in the subgraph
            non_subgraph_col: str, color for nodes not in the subgraph
            graph_name: name for graph used. Only for file naming purposes.
            param_value: value for parameter used (r for rand_geo, p for erdos_renyi, d for rand_regular). Only for file naming purposes.
    Returns: none, plots the MCN trans-subgraph selection of order N, highlighted within G
    """
    PF_vec, arg_sorted_PF_vec = get_centrality(G)

    nodes=set()
    nodes.add(arg_sorted_PF_vec[0])

    for num_nodes in range(2, N+1):
        G0, nodes = get_subgraph(num_nodes, nodes, G, PF_vec, arg_sorted_PF_vec)
        if num_nodes == N: # Mark nodes in subgraph with 1 in their node attribute to indicate they are in that subgraph order.
            nx.set_node_attributes(G, 0, 'subgraph')
            for node in nodes:
                G.nodes[node]['subgraph'] = 1
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), dpi=1000)
    if pos_attr == "pos": 
        pos = nx.get_node_attributes(G, pos_attr)
    elif pos_attr == "circle":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G)
    # Identify nodes within subgraph using node attributes.
    subgraph_indicators = nx.get_node_attributes(G, 'subgraph')
    subgraph_nodes = [node for (node,ind) in subgraph_indicators.items() if ind == 1]
    non_subgraph_nodes = [node for (node,ind) in subgraph_indicators.items() if ind == 0]
    # Create trans-subgraph.
    G0 = G.subgraph(subgraph_nodes)
    G_transed = add_trans_closure_edges(G, G0, 1)

    # Unenhanced subgraph in first subplot.
    nx.draw_networkx_nodes(G, pos, nodelist=subgraph_nodes, ax=axs[0], node_color=subgraph_col, node_size=50, alpha=1, linewidths=0.5)
    nx.draw_networkx_nodes(G, pos, nodelist=non_subgraph_nodes, ax=axs[0], node_color=non_subgraph_col, node_size=50, alpha=0.8, linewidths=0.5) 
    nx.draw_networkx_edges(G, pos, ax=axs[0], width=0.4)
    # Trans-subgraph in second subplot.
    nx.draw_networkx_nodes(G_transed, pos, nodelist=subgraph_nodes, ax=axs[1], node_color=subgraph_col, node_size=50, alpha=1, linewidths=0.5)
    nx.draw_networkx_nodes(G_transed, pos, nodelist=non_subgraph_nodes, ax=axs[1], node_color=non_subgraph_col, node_size=50, alpha=0.8, linewidths=0.5) 
    nx.draw_networkx_edges(G_transed, pos, ax=axs[1], width=0.4)
    
    plt.savefig(f"./graphics/subgraphs_trans_{graph_name}_N{G.number_of_nodes()}_P{param_value}_plot{N}.png")   
    plt.clf()



if __name__=="__main__":
    rand_geo_N100 = nx.random_geometric_graph(100, 0.2, seed=1234)
    rand_er_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/rand_er_N100_P01_S1234.txt", nodetype=int), ordering='default')
    barb_N100 = nx.convert_node_labels_to_integers(nx.read_adjlist("./networks/barbell_N100_P25.txt", nodetype=int), ordering='default')
    wreath_N100_P10 = generate_cluster_wreath(10, 10)

    # draw_topology(rand_geo_N100, pos_attr='pos', node_color='deepskyblue', graph_name="rand_geo", param_value="02")
    # draw_topology(rand_er_N100, pos_attr = "circle", node_color='gold', graph_name='erdos_renyi', param_value='01')
    #draw_topology(barb_N100, node_color='firebrick', graph_name='barbell', param_value='25')
    # draw_topology(wreath_N100_P10, node_color='forestgreen', graph_name='wreath', param_value='10')

    # draw_barbell(barb_N100, M1=25, M2=50, node_color="firebrick", graph_name="barbell", param_value="25")

    # draw_subgraph_choices(barb_N100, ns=[25], pos_attr="barbell", subgraph_col='red', non_subgraph_col='firebrick', M1=25, M2=50, graph_name='barbell', param_value='25')
    #draw_subgraph_choices(rand_geo_N100, ns=[14], pos_attr='pos', subgraph_col='navy', non_subgraph_col='deepskyblue', graph_name='rand_geo', param_value='02')
    #draw_subgraph_choices(rand_geo_N100, ns=[16], pos_attr='pos', subgraph_col='navy', non_subgraph_col='deepskyblue', graph_name='rand_geo', param_value='02')
    #draw_subgraph_choices(wreath_N100_P10, ns=[10], subgraph_col='forestgreen', non_subgraph_col='greenyellow', graph_name='wreath', param_value='10')

    #wreath_N256_P4 = generate_cluster_wreath(64, 4)
    #draw_trans_subgraph(wreath_N256_P4, 200, subgraph_col='forestgreen', non_subgraph_col='greenyellow', graph_name='wreath', param_value='4')

    
    