from .utils import *
from .graphs import *
import os
import matplotlib.pyplot as plt
import networkit as nk



## Saving Figures


def savefig(filename, *dirs, **kwargs):
    """Save a figure with the given filename and directories"""
    thedir = "."
    for dir in dirs:
        thedir = os.path.join(thedir, dir)
        if not os.path.exists(thedir):
            os.makedirs(thedir)
    plt.savefig(os.path.join(thedir, filename), **kwargs)
    LOG(f"Saved figure to {thedir}/{filename}")


## Plotting for matrices


def plot_eigenvalues(M: np.ndarray, fig=None):

    if fig is None:
        fig = plt.figure()

    l, X = np.linalg.eig(M)
    plt.plot(l.real, l.imag, "ro")
    plt.axvline(0, color="k", linewidth=0.5)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()


def plot_gerschgorin_circles(M: np.ndarray, fig=None):

    if fig is None:
        fig = plt.figure()

    ajj = np.diag(M, k=0)
    Rj = np.sum(np.abs(M), axis=1) - np.abs(ajj)

    ax = plt.gca()

    for a, R in zip(ajj, Rj):
        print("drawing circle")
        ax.add_artist(plt.Circle((a.real, a.imag), R, fill=False))

    plt.tight_layout()


## Plotting for graphs


def plot_graph(G: nk.Graph, fig=None, show_labels=False, linewidth=0.5, shape="circle", **annotation_kwargs):
    """Plot a graph"""
    if fig is None:
        fig = plt.figure()

    N = G.numberOfNodes()

    if shape == "circle":
        thetas = np.pi - np.linspace(0, np.pi * 2, N + 1)[:-1]
        rs = thetas * 0 + 1
        xs = rs * np.cos(thetas)
        ys = rs * np.sin(thetas)
    elif shape == "random":
        xs = np.random.uniform(0, 1, N)
        ys = np.random.uniform(0, 1, N)
    else:
        raise ValueError("Shape {shape} is not currently supported")

    # Draw the links
    for u in range(N):
        for v in G.iterNeighbors(u):
            if v > u:
                plt.plot([xs[u], xs[v]], [ys[u], ys[v]], "-k", linewidth=linewidth, zorder=1)

    # Draw nodes
    plt.scatter(xs, ys, color="r", edgecolor="k", marker="o", zorder=1000, s=100)
    r_margin = 1.1
    if show_labels:
        for i, (x, y) in enumerate(zip(xs, ys)):
            plt.text(r_margin * x, r_margin * y, str(i), ha="center", va="center")

    text = ""
    space = 20
    format = "10.5f"

    for key, value in annotation_kwargs.items():
        text += f"{key:{space}s} : {value:{format}}\n"

    plt.axis("equal")
    plt.grid(False)
    plt.axis("off")
    plt.tight_layout()
    plt.text(1.1, 0, text, ha="left", va="top", transform=plt.gca().transAxes)
    
    
def plot_graph_spectrum(graph_func:callable, N=10, *graph_args, label=None, N_realisations=100, **graph_kwargs):
    """Plot the spectrum of a graph"""

    eigs = np.array([])
    for i in range(N_realisations):
        G = graph_func(N=N, *graph_args, **graph_kwargs)
        A = adjacency_matrix(G)
        l, _ = np.linalg.eig(A)
        l = l.real
        eigs = np.append(eigs, l)

    tops, bin_edges = np.histogram(eigs, bins=N, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(centers, tops, label=label)
    plt.tight_layout()


def plot_degrees(G: nk.Graph, fig=None, **kwargs):

    if fig is None:
        fig = plt.figure()

    degrees = degrees_G(G)
    plt.hist(degrees, color="r", edgecolor="k", **kwargs)

    N = G.numberOfNodes()
    L = G.numberOfEdges()
    plt.axvline(2 * L / N, color="y", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()


def plot_degree_hist_distribution(degree_history, *args, **kwargs):
    """Plot the degree distribution of a graph"""
    degree_history.sort()
    plt.figure()
    plt.hist(degree_history, density=True, *args, **kwargs)

    mu, sigma = stats.norm.fit(degree_history)
    pdf = stats.norm.pdf(degree_history, mu, sigma)
    plt.plot(degree_history, pdf, "c-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree distribution")


def plot_degree_distribution(G: nk.Graph, *args, **kwargs):
    """Plot the degree distribution of a graph"""
    plt.figure()
    degrees = G.degrees()
    plt.hist(degrees, density=True, *args, **kwargs)

    mu, sigma = stats.norm.fit(degrees)
    pdf = stats.norm.pdf(degrees, mu, sigma)
    plt.plot(degrees, pdf, "r-")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree distribution")
