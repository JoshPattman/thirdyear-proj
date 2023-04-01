import numpy as np
import math

def count_connections(n):
    if n == 1:
        return 0
    return count_connections(n-1)+n-1

def is_connection_in(c, cons):
    return c in cons or (c[1], c[0]) in cons

def random_pair(length):
    a = np.random.randint(0, length)
    b = a
    while b == a:
        b = np.random.randint(0, length)
    return a, b

def fully_connected_graph(nodes_names, density=0):
    # At 0 density the graph is connected minimally (all nodes connect to all nodes)
    # At 1 density the graph is fully connected
    nodes = [[x] for x in nodes_names]
    # Make a random fully connected web
    connections = []
    while len(nodes) > 1:
        # Pick the wto random groups. They cannot be the same group
        a, b = random_pair(len(nodes))
        na, nb = nodes[a], nodes[b]
        # Pick the member of each group to connect
        ma, mb = na[np.random.randint(0, len(na))], nb[np.random.randint(0, len(nb))]
        connections.append((ma, mb))

        newgroup = []
        for n in na:
            newgroup.append(n)
        for n in nb:
            newgroup.append(n)
        nodes = [n for n in nodes if n not in [nodes[a], nodes[b]]]
        nodes.append(newgroup)
    # Add density
    minimum_connections = len(connections)
    maximum_connections = count_connections(len(nodes_names))
    target_connections = int((maximum_connections - minimum_connections)*density)+minimum_connections
    while len(connections) < target_connections:
        con = connections[0]
        while is_connection_in(con, connections):
            a, b = random_pair(len(nodes_names))
            con = (nodes_names[a], nodes_names[b])
        connections.append(con)
    return connections

def mean_min_steps(nodes, connections):
    # Initialize the distances matrix with infinity for all pairs of nodes
    distances = [[math.inf for _ in range(len(nodes))] for _ in range(len(nodes))]

    # Set the distances for adjacent nodes to their edge weights
    for u, v in connections:
        i = nodes.index(u)
        j = nodes.index(v)
        distances[i][j] = 1
        distances[j][i] = 1

    # Floyd-Warshall algorithm for finding the shortest path distances between every pair of nodes
    for k in range(len(nodes)):
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])

    # Calculate the mean minimum number of steps
    total = 0
    count = 0
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            total += distances[i][j]
            count += 1

    mean_min = total / count
    return mean_min

def mean_connections_per_node(nodes, connections):
    return len(connections) / len(nodes)