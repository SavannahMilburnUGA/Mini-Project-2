import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Node states
BELIEVER = 0
NON_BELIEVER = 1
NEUTRAL = 2

# -----------------------------
# Setup and Simulation Functions
# -----------------------------

def initialize_states(G, pb0=0.3, pn0=0.3, seed=42):
    random.seed(seed)
    states = {}
    n_nodes = G.number_of_nodes()
    n_believers = int(pb0 * n_nodes)
    n_non_believers = int(pn0 * n_nodes)
    n_neutrals = n_nodes - n_believers - n_non_believers
    all_states = [BELIEVER]*n_believers + [NON_BELIEVER]*n_non_believers + [NEUTRAL]*n_neutrals
    random.shuffle(all_states)
    for i, node in enumerate(G.nodes()):
        states[node] = all_states[i]
    return states

def get_neighbor_counts(G, states, node):
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return 0, 0, 0
    counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
    for n in neighbors:
        counts[states[n]] += 1
    total = len(neighbors)
    return counts[BELIEVER]/total, counts[NON_BELIEVER]/total, counts[NEUTRAL]/total

def calculate_local_dynamics(pb, pn, pnb=0.1, pbn=0.05, r=1.0):
    pneu = 1 - pb - pn
    dpb = pnb * pn - pbn * pb + r * max(pb - pn, 0) * pneu - r * max(pn - pb, 0) * pb
    dpn = pbn * pb - pnb * pn + r * max(pn - pb, 0) * pneu - r * max(pb - pn, 0) * pn
    return dpb, dpn, -(dpb + dpn)

def calculate_transition_probabilities(pb, pn, pnb=0.1, pbn=0.05, r=1.0):
    dpb, dpn, dpneu = calculate_local_dynamics(pb, pn, pnb, pbn, r)
    prob_b = max(0, min(1, 0.33 + 0.1 * dpb))
    prob_n = max(0, min(1, 0.33 + 0.1 * dpn))
    prob_neu = max(0, min(1, 0.33 + 0.1 * dpneu))
    total = prob_b + prob_n + prob_neu
    return prob_b/total, prob_n/total, prob_neu/total

def update_node_state(current, prob_b, prob_n, prob_neu):
    r = random.random()
    if r < prob_b:
        return BELIEVER
    elif r < prob_b + prob_n:
        return NON_BELIEVER
    return NEUTRAL

def simulate_network(G, states, T=50, pnb=0.1, pbn=0.05, r=0.5):
    history = []
    global_stats = []
    for _ in range(T):
        history.append(states.copy())
        counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
        for s in states.values():
            counts[s] += 1
        N = len(states)
        global_stats.append({
            'pb': counts[BELIEVER] / N,
            'pn': counts[NON_BELIEVER] / N,
            'pneutral': counts[NEUTRAL] / N
        })
        new_states = {}
        for node in G.nodes():
            pb, pn, pneu = get_neighbor_counts(G, states, node)
            prob_b, prob_n, prob_neu = calculate_transition_probabilities(pb, pn, pnb, pbn, r)
            new_states[node] = update_node_state(states[node], prob_b, prob_n, prob_neu)
        states = new_states
    return history, global_stats

# -----------------------------
# Visualization
# -----------------------------

def create_graph(graph_type, N=100, radius=0.3):
    if graph_type == "geometric":
        G = nx.random_geometric_graph(N, radius)
        pos = nx.get_node_attributes(G, "pos")
    elif graph_type == "random":
        G = nx.erdos_renyi_graph(N, 0.1)
        pos = nx.spring_layout(G)
    elif graph_type == "small_world":
        G = nx.watts_strogatz_graph(N, 4, 0.2)
        pos = nx.spring_layout(G)
    elif graph_type == "scale_free":
        G = nx.barabasi_albert_graph(N, 2)
        pos = nx.spring_layout(G)
    else:
        raise ValueError("Unknown graph type")
    return G, pos

def visualize_all(G, pos, hist, stats, graph_type):
    color_map = {0: 'blue', 1: 'red', 2: 'gray'}
    steps = [0, len(hist)//3, 2*len(hist)//3, len(hist)-1]
    fig, axes = plt.subplots(2, len(steps), figsize=(5*len(steps), 8))

    # Network snapshots
    for i, t in enumerate(steps):
        ax = axes[0, i]
        colors = [color_map[hist[t][n]] for n in G.nodes()]
        nx.draw(G, pos, node_color=colors, node_size=80, ax=ax)
        ax.set_title(f"{graph_type} t={t}")
        ax.axis('off')

    # Global stats plot
    ax2 = axes[1, 0]
    pb = [s['pb'] for s in stats]
    pn = [s['pn'] for s in stats]
    pneu = [s['pneutral'] for s in stats]
    ax2.plot(pb, label='Believers', color='blue')
    ax2.plot(pn, label='Non-believers', color='red')
    ax2.plot(pneu, label='Neutral', color='gray')
    ax2.set_title("Global Belief Proportions")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Proportion")
    ax2.legend()
    ax2.grid(True)

    # Hide extra subplots
    for j in range(1, len(steps)):
        axes[1, j].axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Run All Networks
# -----------------------------
for graph_type in ["geometric", "random", "small_world", "scale_free"]:
    G, pos = create_graph(graph_type, N=100)
    states = initialize_states(G)
    history, stats = simulate_network(G, states, T=40)
    visualize_all(G, pos, history, stats, graph_type)
