import networkit as nk
import numpy as np
import random


## Graph Matrices


def adjacency_matrix(G: nk.Graph):
    """Return the adjacency matrix of a graph"""
    A = np.zeros((G.numberOfNodes(), G.numberOfNodes()))
    for u in range(G.numberOfNodes()):
        for v in G.iterNeighbors(u):
            A[u, v] = 1
    return A


def incidence_matrix(G: nk.Graph):
    """Return the incidence matrix of a graph"""
    B = np.zeros((G.numberOfNodes(), G.numberOfEdges()))
    for u in range(G.numberOfNodes()):
        for i, v in enumerate(G.iterNeighbors(u)):
            B[u, i] = 1
    return B


def laplacian_matrix(G: nk.Graph):
    """Return the laplacian matrix of a graph"""
    A = adjacency_matrix(G)
    D = np.diag(np.sum(A, axis=0))
    return D - A


def degree_matrix(G: nk.Graph):
    """Return the degree matrix of a graph"""
    A = adjacency_matrix(G)
    D = np.diag(np.sum(A, axis=0))
    return D


def degrees_G(G: nk.Graph):
    return np.array([G.degree(i) for i in G.iterNodes()])


def hopcount_matrix_diameter(G: nk.Graph):
    """Calculate the hopcount between two nodes in a graph"""
    A = adjacency_matrix(G)
    I = np.identity(G.numberOfNodes())
    Ak = A.copy()
    H = np.zeros_like(A)
    k = 1
    Asum = I + A
    while not (Asum != 0).all():
        H[(Ak != 0) * (H == 0)] = k
        k += 1
        Ak = Ak.dot(A)
        Asum += Ak
    H[I.astype(bool)] = 0
    rho = max(k, 1)

    return H.astype(int), rho

def pseudo_inverse(Q:np.ndarray):
    J = np.ones_like(Q)
    a = 1
    N = Q.shape[0]
    Qd = np.linalg.inv(Q + a * J) - J / (a * N * N) 
    return Qd

def effective_resistance(Qd: np.ndarray):
    zeta = np.diag(Qd)[np.newaxis].T
    u = np.ones_like(zeta)
    Omega = u.dot(zeta.T) + zeta.dot(u.T) - 2 * Qd
    return Omega


## Graph Operations


def line_graph(G: nk.Graph):
    """Return the line graph of a graph"""
    L = G.numberOfEdges()
    lG = nk.Graph(L)

    for i, u in enumerate(G.iterEdges()):
        for j, v in enumerate(G.iterEdges()):
            if i < j and (u[0] in v or u[1] in v):
                lG.addEdge(i, j)
    return lG


def complement_graph(G: nk.Graph):
    """Return the complement of a graph"""
    N = G.numberOfNodes()
    cG = nk.Graph(N)
    for u in range(N):
        for v in range(u + 1, N):
            if not G.hasEdge(u, v):
                cG.addEdge(u, v)
    return cG


def graph_union(G1: nk.Graph, G2: nk.Graph):
    N = G1.numberOfNodes()
    if N != G2.numberOfNodes():
        raise ValueError("Graphs must have the same number of nodes")

    G_union = nk.Graph(N)
    for u in range(N):
        for v in G1.iterNeighbors(u):
            if not G_union.hasEdge(u, v):
                G_union.addEdge(u, v)
        for v in G2.iterNeighbors(u):
            if not G_union.hasEdge(u, v):
                G_union.addEdge(u, v)

    return G_union


## Graph Metrics


def avg_degree(G):
    """Calculate the average degree of a graph"""
    return 2 * G.numberOfEdges() / G.numberOfNodes()


def spectral_radius(G):
    """Calculate the spectral radius of a graph"""
    A = adjacency_matrix(G)
    l, X = np.linalg.eig(A)
    l.sort()
    return l[-1]


def walks(G: nk.Graph, k: int):
    A = adjacency_matrix(G)
    Ak = A.copy()
    for i in range(k - 1):
        print(i)
        Ak = Ak.dot(A)
    return np.trace(Ak)


def clustering_coefficient(G: nk.Graph):
    """Calculate the clustering coefficient of a graph"""
    W3 = walks(G, 3)
    d = degrees_G(G)
    L = G.numberOfEdges()
    return W3 / (d.dot(d) - 2 * L)


def average_hopcount(G: nk.Graph):
    """Calculate the average hopcount of a graph"""
    H, _ = hopcount_matrix_diameter(G)
    N = G.numberOfNodes()
    return H.sum() / (N * (N - 1))


def algebraic_connectivity(G: nk.Graph):
    """Calculate the algebraic connectivity of a graph"""
    Q = laplacian_matrix(G)
    l, X = np.linalg.eig(Q)
    l.sort()
    return l[1]

def effective_resistance_G(G: nk.Graph):
    Q = laplacian_matrix(G)
    Qd = pseudo_inverse(Q)
    return effective_resistance(Qd)


def is_connected(G: nk.Graph):
    return bool(algebraic_connectivity(G) > 0)


## Standard Graphs


def example_graph():
    """Graph used in the lecture slides as example"""
    G = nk.Graph(6)
    G.addEdge(0, 1)
    G.addEdge(0, 2)
    G.addEdge(0, 5)
    G.addEdge(1, 2)
    G.addEdge(1, 4)
    G.addEdge(1, 5)
    G.addEdge(2, 3)
    G.addEdge(3, 4)
    G.addEdge(4, 5)
    return G


def complete_graph(N: int):
    """Return a complete graph with N nodes"""
    G = nk.Graph(N)
    for u in range(N):
        for v in range(u + 1, N):
            if not G.hasEdge(u, v):
                G.addEdge(u, v)
    return G


def star_graph(N: int):
    """Return a star graph with N nodes"""
    G = nk.Graph(N)
    for u in range(1, N):
        G.addEdge(0, u)
    return G


def ring_graph(N: int):
    """Return a ring graph with N nodes"""
    G = nk.Graph(N)
    for u in range(N):
        G.addEdge(u, (u + 1) % N)
    return G


def ring_lattice_graph(N: int, k: int):
    """Return a ring lattice graph with N nodes and degree k"""
    G = nk.Graph(N)
    for u in range(N):
        for i in range(k):
            v = (u + i + 1) % N
            if not G.hasEdge(u, v):
                G.addEdge(u, v)
    return G


def small_world_graph(N: int, k: int, p: float):
    """Return a small world graph with N nodes, degree k, and rewiring probability p"""
    G = ring_lattice_graph(N, k)
    for u in range(N):
        for v in G.iterNeighbors(u):
            if random.random() < p:
                G.removeEdge(u, v)
                w = random.randint(0, N - 1)
                while G.hasEdge(u, w) or w == u:
                    w = random.randint(0, N - 1)
                G.addEdge(u, w)
    return G


def star_graph(N: int):
    """Return a star graph with N nodes"""
    G = nk.Graph(N)
    for u in range(N):
        v = (u + 2) % N
        if not G.hasEdge(u, v):
            G.addEdge(u, v)
    return G


def heart_graph(N: int):
    G = nk.Graph(N)
    for u in range(N):
        G.addEdge(u, (2 * u) % N)
    return G


def regular_graph(N: int, r: int):
    """Return a regular graph with N nodes and degree r"""
    G = nk.Graph(N)
    for u in range(N):
        for i in range(r):
            v = (u + i + 1) % N
            G.addEdge(u, v)
    return G


def turan_graph(N: int, r: int):
    """Return a Turan graph with N nodes and r cliques"""
    G = nk.Graph(N)
    N = G.numberOfNodes()
    nodes = np.array(list(G.iterNodes()))
    subsets = np.array_split(nodes, 4)
    for ss1 in subsets:
        for ss2 in subsets:
            if not ss1 is ss2:
                for u in ss1:
                    for v in ss2:
                        if not G.hasEdge(u, v):
                            G.addEdge(u, v)
    return G


## Random Graphs


def erdos_renyi_graph(N: int, p: float):
    """Return a Erdos-Renyi graph with N nodes and probability p"""
    G = nk.Graph(N)
    for i in range(N):
        for j in range(i):
            if random.random() < p:
                G.addEdge(i, j)
    return G


def erdos_renyi_graph_L(N: int, L: int):
    """Return a Erdos-Renyi graph with N nodes and L edges"""
    G = nk.Graph(N)
    for i in range(L):
        u = v = 0
        while u == v or G.hasEdge(u, v):
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
        G.addEdge(u, v)
    return G


def barabasi_albert_graph(N: int, m: int):
    """Return a Barabasi-Albert graph with N nodes and m edges"""
    G = nk.Graph(N)
    for i in range(m):
        G.addEdge(i, i + 1)
    for i in range(m, N):
        for j in range(m):
            G.addEdge(i, random.randint(0, i - 1))
    return G
