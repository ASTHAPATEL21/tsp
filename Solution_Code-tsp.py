import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Function to calculate the Euclidean distance between two points
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Nearest Neighbor Algorithm for TSP
def nearest_neighbor(graph, start_node, nodes):
    unvisited_nodes = set(graph.nodes)
    current_node = start_node
    path = [current_node]

    while unvisited_nodes:
        next_node = min(unvisited_nodes)
        path.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node
        plot_tsp_tour(graph, nodes, path)

    return path

# Simulation Environment
def simulate_tsp(num_nodes, seed=None):
    np.random.seed(seed)
    nodes = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}

    # Create a complete graph with Euclidean distances as weights
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            graph.add_edge(i, j, weight=distance(nodes[i], nodes[j]))

    return nodes, graph

# Plot the TSP tour
def plot_tsp_tour(graph, nodes, tour):
    tour_edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)] + [(tour[-1], tour[0])]
    pos = {i: nodes[i] for i in nodes}

    nx.draw_networkx_nodes(graph, pos, node_size=200)
    # nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edges(graph, pos, tour_edges, edge_color='r', width=2)
    nx.draw_networkx_labels(graph, pos)

    plt.title("TSP Tour")
    plt.show()

# Main function
def main():
    # Simulation parameters
    num_nodes = 7
    start_node = 0

    # Simulate TSP
    nodes, graph = simulate_tsp(num_nodes, seed=42)

    # Nearest Neighbor Algorithm
    tour_nn = nearest_neighbor(graph, start_node, nodes)

    # Plot the results
    plot_tsp_tour(graph, nodes, tour_nn)


main()