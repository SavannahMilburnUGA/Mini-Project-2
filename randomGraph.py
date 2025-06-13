import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Node states
BELIEVER = 0
NON_BELIEVER = 1  
NEUTRAL = 2       

# Create random geometric graph: place N nodes randomly in [0, 1]x[0, 1] square, connect any 2 nodes whose Euclidean distance ≤ radius: spatial clustering
# Uses default values
def create_spatial_graph(n_nodes=20, radius=0.3, seed=42):
    # Set random seed for reproducible node positions and connections ???
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    G = nx.random_geometric_graph(n_nodes, radius, seed=seed)
    # Store node positions for visualization and analysis - each node's location are (x,y) coordinates
    pos = nx.get_node_attributes(G, 'pos')
    return G, pos

# Generate random spanning tree from the graph for random walker
# Does not return a graph if graph is not connected
def generate_random_spanning_tree(G, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Check if graph is connected - MUST be connected for walker to visit all nodes
    if not nx.is_connected(G):
        print(f"ERROR: Graph is not fully connected!")
        print(f"Graph has {nx.number_connected_components(G)} connected components.")
        print(f"Walker cannot visit all nodes. Please increase the connection radius or reduce number of nodes.")
        return None
    
    # Use NetworkX's built-in random spanning tree (uses Wilson's algorithm internally)
    spanning_tree = nx.random_spanning_tree(G, seed=seed)
    
    return spanning_tree

# Random walker that visits all nodes using random spanning tree
# Returns walker_path: sequence of nodes visited + coverage_states for measuring efficiency
def random_walk_coverage(G, spanning_tree, start_node=None, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Choose random starting node if not specified
    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    
    # Initialize tracking variables
    walker_path = [start_node]
    visited_nodes = {start_node}
    current_node = start_node
    
    # Efficiency tracking
    revisit_counts = defaultdict(int)
    revisit_counts[start_node] = 1
    coverage_times = {start_node: 0}
    spatial_distances = []
    step_count = 0
    
    # Get node positions for spatial distance calculations
    pos = nx.get_node_attributes(G, 'pos')
    
    # Continue until all nodes are visited
    total_nodes = G.number_of_nodes()
    
    while len(visited_nodes) < total_nodes:
        # Get all neighbors of current node in the original graph
        neighbors = list(G.neighbors(current_node))
        
        if not neighbors:
            print(f"Warning: Node {current_node} has no neighbors. This shouldn't happen in a connected graph.")
            break
        
        # Choose next node randomly from neighbors
        next_node = random.choice(neighbors)
        
        # Record the move
        walker_path.append(next_node)
        step_count += 1
        
        # Calculate spatial distance if positions are available
        if pos and current_node in pos and next_node in pos:
            dist = np.sqrt((pos[current_node][0] - pos[next_node][0])**2 + 
                          (pos[current_node][1] - pos[next_node][1])**2)
            spatial_distances.append(dist)
        
        # Update tracking
        revisit_counts[next_node] += 1
        
        # Record first visit time
        if next_node not in visited_nodes:
            visited_nodes.add(next_node)
            coverage_times[next_node] = step_count
        
        current_node = next_node
    
    # Calculate efficiency metrics
    coverage_stats = calculate_walker_efficiency(walker_path, coverage_times, revisit_counts, 
                                               spatial_distances, total_nodes)
    
    return walker_path, coverage_stats

# Calculate random walker efficiency metrics
def calculate_walker_efficiency(walker_path, coverage_times, revisit_counts, spatial_distances, total_nodes):
    stats = {}
    
    # Basic coverage metrics
    stats['total_steps'] = len(walker_path) - 1  # Subtract 1 because path includes starting position
    stats['steps_to_full_coverage'] = max(coverage_times.values()) if coverage_times else 0
    stats['nodes_covered'] = len(coverage_times)
    stats['coverage_ratio'] = stats['nodes_covered'] / total_nodes
    
    # Efficiency ratios
    if stats['steps_to_full_coverage'] > 0:
        stats['efficiency_ratio'] = (total_nodes - 1) / stats['steps_to_full_coverage']  # Ideal vs actual
    else:
        stats['efficiency_ratio'] = 1.0
    
    # Revisit analysis
    total_revisits = sum(count - 1 for count in revisit_counts.values() if count > 1)
    stats['total_revisits'] = total_revisits
    stats['backtracking_ratio'] = total_revisits / stats['total_steps'] if stats['total_steps'] > 0 else 0
    
    # Spatial efficiency (if spatial distances available)
    if spatial_distances:
        stats['total_spatial_distance'] = sum(spatial_distances)
        stats['average_step_distance'] = np.mean(spatial_distances)
        stats['spatial_efficiency'] = len(spatial_distances) / stats['total_spatial_distance'] if stats['total_spatial_distance'] > 0 else 0
    
    # Coverage time statistics
    if coverage_times:
        coverage_time_values = list(coverage_times.values())
        stats['mean_coverage_time'] = np.mean(coverage_time_values)
        stats['std_coverage_time'] = np.std(coverage_time_values)
        stats['max_coverage_time'] = max(coverage_time_values)
    
    return stats

# Initialize node states across network G w/ initial believer prop, initial non-believer prop, seed
# Returns dictionary mapping each node to its initial state ???
def initialize_states(G, pb0=0.3, pn0=0.3, seed=42):
    if seed is not None:
        random.seed(seed)
    
    states = {}
    n_nodes = G.number_of_nodes()
    
    # Calculate exact # of nodes for each state
    n_believers = int(pb0 * n_nodes)
    n_non_believers = int(pn0 * n_nodes)
    n_neutrals = n_nodes - n_believers - n_non_believers  # Remainder are neutral
    
    # Create list of all initial states
    all_states = ([BELIEVER] * n_believers + [NON_BELIEVER] * n_non_believers + [NEUTRAL] * n_neutrals)
    
    # Randomly shuffle states + assign to nodes for random spatial distribution
    random.shuffle(all_states)
    for i, node in enumerate(G.nodes()):
        states[node] = all_states[i]
    
    return states

# Compute local info environment for specific node = find neighbors (stronger influence)
# G is network, states is current info states of all nodes, node is specific node we are looking for neighbors
# Returns proportions of each state among neighbors
def get_neighbor_counts(G, states, node):
    # Get all spatially connected neighbors via nodes w/in connection radius
    neighbors = list(G.neighbors(node))
    
    # Nodes w/ no links - no influence
    if not neighbors:
        return 0, 0, 0
    
    # Count information states among spatial neighbors
    counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
    for neighbor in neighbors:
        counts[states[neighbor]] += 1
    
    # Convert to proportions for the dynamic network model equations - local info environment for each node
    total = len(neighbors)
    pb = counts[BELIEVER] / total      # Proportion of believing neighbors
    pn = counts[NON_BELIEVER] / total  # Proportion of non-believing neighbors  
    pneutral = counts[NEUTRAL] / total # Proportion of neutral neighbors
    
    return pb, pn, pneutral

# Apply dynamic network model equations to each node based on its neighbors
# pb_local: proportion believers out of neighbors, likewise for pn_local
# r = strenght of spatial peer pressure ???
# Returns change in proportions (delta_pb, delta_pn, delta_pneutral)
def calculate_local_dynamics(pb_local, pn_local, pnb=0.1, pbn=0.05, r=1.0):
    # Calculate neutral proportion (conservation constraint)
    pneutral_local = 1 - pb_local - pn_local
    
    # BELIEVER DYNAMICS
    # Fixed conversion: constant rate of opinion change regardless of spatial context
    fixed_b_growth = pnb * pn_local - pbn * pb_local
    
    # Majority influence: when believers dominate spatially, they recruit neutrals
    # max(pb_local - pn_local, 0) ensures this only happens when believers > non-believers
    # Multiplied by pneutral_local: can only recruit existing neutrals
    majority_b_growth = r * max(pb_local - pn_local, 0) * pneutral_local
    
    # Minority pressure: when non-believers dominate spatially, believers become neutral
    # max(pn_local - pb_local, 0) ensures this only happens when non-believers > believers  
    # Multiplied by pb_local: pressure is proportional to believer population
    pressure_b_loss = r * max(pn_local - pb_local, 0) * pb_local
    
    # Net change in believer proportion
    delta_pb = fixed_b_growth + majority_b_growth - pressure_b_loss
    
    # NON-BELIEVER DYNAMICS (symmetric to believers)
    fixed_n_growth = pbn * pb_local - pnb * pn_local
    majority_n_growth = r * max(pn_local - pb_local, 0) * pneutral_local  
    pressure_n_loss = r * max(pb_local - pn_local, 0) * pn_local
    
    delta_pn = fixed_n_growth + majority_n_growth - pressure_n_loss
    
    # NEUTRAL DYNAMICS (conservation: total proportion = 1)
    delta_pneutral = -(delta_pb + delta_pn)
    
    return delta_pb, delta_pn, delta_pneutral

# Convert deterministic dynamics into stochastic transition probabilities (random individual decisions)
# Returns probabilities of switching 
def calculate_transition_probabilities(pb_local, pn_local, pnb=0.1, pbn=0.05, r=1.0):
    # Get deterministic changes from spatial dynamics
    delta_pb, delta_pn, delta_pneutral = calculate_local_dynamics(pb_local, pn_local, pnb, pbn, r)
    
    # Convert to probabilities with scaling factor: prob_factor controls how quickly spatial influence affects individual decisions
    prob_factor = 0.1  
    prob_to_b = max(0, min(1, 0.33 + prob_factor * delta_pb))
    prob_to_n = max(0, min(1, 0.33 + prob_factor * delta_pn))
    prob_to_neutral = max(0, min(1, 0.33 + prob_factor * delta_pneutral))
    
    # Normalize to ensure valid probability distribution
    total = prob_to_b + prob_to_n + prob_to_neutral
    if total > 0:
        prob_to_b /= total
        prob_to_n /= total
        prob_to_neutral /= total
    else:
        # Fallback: equal probabilities
        prob_to_b = prob_to_n = prob_to_neutral = 1/3
    
    return prob_to_b, prob_to_n, prob_to_neutral

# Stochastically update a certain node's info state - each node makes probabilistic choice based on spatial influence
def update_node_state(current_state, prob_to_b, prob_to_n, prob_to_neutral):
    rand = random.random()
    
    # Stochastic state transition based on calculated probabilities
    if rand < prob_to_b:
        return BELIEVER
    elif rand < prob_to_b + prob_to_n:
        return NON_BELIEVER
    else:
        return NEUTRAL

# Simulate spatial info spread over time - main simulation loop
# Takes in network G, initial states, # of time steps, + proportions w/ rate
# Returns info states at each time step + proportions
def simulate_network(G, initial_states, T=50, pnb=0.1, pbn=0.05, r=1.0):
    # Storage for tracking information spread over time
    states_history = []  # Local states at each node
    global_stats = []    # Global information proportions
    
    states = initial_states.copy()
    
    # Main simulation loop - each iteration represents one time step
    for t in range(T):
        # Record current information distribution
        states_history.append(states.copy())
        
        # Calculate global information statistics
        n_nodes = len(states)
        counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
        for state in states.values():
            counts[state] += 1
        
        global_stats.append({
            'pb': counts[BELIEVER] / n_nodes,
            'pn': counts[NON_BELIEVER] / n_nodes,
            'pneutral': counts[NEUTRAL] / n_nodes
        })
        
        # Calculate new states for all nodes simultaneously - parallel info processing across network ?
        new_states = {}
        
        for node in G.nodes():
            # Get local spatial information environment
            pb_local, pn_local, pneutral_local = get_neighbor_counts(G, states, node)
            
            # Calculate how spatial neighbors influence this node
            prob_to_b, prob_to_n, prob_to_neutral = calculate_transition_probabilities(
                pb_local, pn_local, pnb, pbn, r
            )
            
            # Update node's information state based on spatial influence
            new_states[node] = update_node_state(
                states[node], prob_to_b, prob_to_n, prob_to_neutral
            )
        
        # Apply all state changes simultaneously (synchronous update) - prevents order effects 
        states = new_states
    
    # Record final state for analysis
    states_history.append(states.copy())
    n_nodes = len(states)
    counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
    for state in states.values():
        counts[state] += 1
    global_stats.append({
        'pb': counts[BELIEVER] / n_nodes,
        'pn': counts[NON_BELIEVER] / n_nodes,
        'pneutral': counts[NEUTRAL] / n_nodes
    })
    
    return states_history, global_stats

# Visualize how info spread thru network over time via spatial patterns (info clusters) + temporal patterns (global proportions)
def visualize_spatial_network_evolution(G, pos, states_history, global_stats, timesteps_to_show=[0, 10, 25, 49], 
                                      walker_path=None, spanning_tree=None, coverage_stats=None):
    colors = {BELIEVER: 'blue', NON_BELIEVER: 'red', NEUTRAL: 'lightgray'}
    labels = {BELIEVER: 'Believer', NON_BELIEVER: 'Non-believer', NEUTRAL: 'Neutral'}
    
    # Adjust subplot layout based on whether we have walker information
    if walker_path is not None:
        fig, axes = plt.subplots(3, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 12))
        if len(timesteps_to_show) == 1:
            axes = axes.reshape(-1, 1)
    else:
        fig, axes = plt.subplots(2, len(timesteps_to_show), figsize=(4*len(timesteps_to_show), 8))
        if len(timesteps_to_show) == 1:
            axes = axes.reshape(-1, 1)
    
    # Spatial network visualization at different time points
    for i, t in enumerate(timesteps_to_show):
        ax = axes[0, i]
        states = states_history[t]
        
        # Color nodes by their information state
        node_colors = [colors[states[node]] for node in G.nodes()]
        
        # Draw network using spatial positions
        nx.draw(G, pos, ax=ax, node_color=node_colors, with_labels=True,
                node_size=300, font_size=8, font_weight='bold', edge_color='gray', alpha=0.7)
        ax.set_title(f'Spatial Network at t={t}')
        ax.set_aspect('equal')
    
    # Show spanning tree if available
    if spanning_tree is not None and walker_path is not None:
        for i, t in enumerate(timesteps_to_show):
            ax = axes[1, i]
            states = states_history[t]
            node_colors = [colors[states[node]] for node in spanning_tree.nodes()]
            
            # Draw spanning tree
            nx.draw(spanning_tree, pos, ax=ax, node_color=node_colors, with_labels=True,
                    node_size=300, font_size=8, font_weight='bold', edge_color='green', 
                    alpha=0.8, width=2)
            ax.set_title(f'Spanning Tree at t={t}')
            ax.set_aspect('equal')
    
    # Global information dynamics over time
    row_idx = 2 if walker_path is not None else 1
    ax = plt.subplot(fig.get_axes()[0].get_gridspec().nrows, 1, row_idx + 1)
    
    times = range(len(global_stats))
    pb_vals = [stats['pb'] for stats in global_stats]
    pn_vals = [stats['pn'] for stats in global_stats]
    pneutral_vals = [stats['pneutral'] for stats in global_stats]
    
    ax.plot(times, pb_vals, 'b-', label='Believers', linewidth=2)
    ax.plot(times, pn_vals, 'r-', label='Non-believers', linewidth=2)
    ax.plot(times, pneutral_vals, 'gray', label='Neutrals', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')
    
    # Add walker efficiency info to title if available
    if coverage_stats:
        title = f'Global Information Dynamics (Walker Efficiency: {coverage_stats["efficiency_ratio"]:.3f})'
    else:
        title = 'Global Information Dynamics'
    ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Allow user to input # of nodes to dynamically change graph
def getUserInput():
    n_nodes = int(input("Enter the number of nodes for the network (e.g., 20, 50, 100): "))
    if n_nodes > 200:
        print("Warning: Large networks (>200 nodes) may take longer to simulate and visualize.")
    return n_nodes
# Print random walker efficiency stats 
def print_walker_efficiency_report(coverage_stats):  # <- This line should start at column 0
    print("\n=== Random Walker Efficiency Report ===")
    print(f"Total steps taken: {coverage_stats['total_steps']}")
    print(f"Steps to full coverage: {coverage_stats['steps_to_full_coverage']}")
    print(f"Nodes covered: {coverage_stats['nodes_covered']}")
    print(f"Coverage ratio: {coverage_stats['coverage_ratio']:.3f}")
    print(f"Efficiency ratio: {coverage_stats['efficiency_ratio']:.3f}")
    print(f"Total revisits: {coverage_stats['total_revisits']}")
    print(f"Backtracking ratio: {coverage_stats['backtracking_ratio']:.3f}")

    if 'total_spatial_distance' in coverage_stats:
        print(f"Total spatial distance: {coverage_stats['total_spatial_distance']:.3f}")
        print(f"Average step distance: {coverage_stats['average_step_distance']:.3f}")
        print(f"Spatial efficiency: {coverage_stats['spatial_efficiency']:.3f}")

    if 'mean_coverage_time' in coverage_stats:
        print(f"Mean coverage time: {coverage_stats['mean_coverage_time']:.2f}")
        print(f"Std coverage time: {coverage_stats['std_coverage_time']:.2f}")
        print(f"Max coverage time: {coverage_stats['max_coverage_time']}") 

# Main
# Example usage demonstrating spatial information spread
if __name__ == "__main__":
    print("=== Spatial Information Spread Model with Random Walker ===")
    print("Modeling spatial influence on dynamics of info spread\n")
    
    # Create spatial network
    n_nodes = getUserInput()
    G, pos = create_spatial_graph(n_nodes=n_nodes, radius=0.3, seed=42)
    
    print(f"Created spatial graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Connection radius: 0.3")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    # Generate random spanning tree and perform random walk
    print(f"\nGenerating random spanning tree and performing random walk...")
    spanning_tree = generate_random_spanning_tree(G, seed=42)
    
    # Only proceed if graph is connected
    if spanning_tree is None:
        print("Cannot proceed with walker simulation. Exiting...")
        exit()
    
    walker_path, coverage_stats = random_walk_coverage(G, spanning_tree, seed=42)
    
    print(f"Spanning tree has {spanning_tree.number_of_nodes()} nodes and {spanning_tree.number_of_edges()} edges")
    print(f"Walker visited {len(set(walker_path))} unique nodes in {len(walker_path)-1} steps")
    
    # Print detailed walker efficiency report
    print_walker_efficiency_report(coverage_stats)
    
    # MODIFIED: 
    # # Get user input for initial proportions
    # pb0 = float(input("Enter initial proportion of believers (e.g., 0.3): "))
    # pn0 = float(input("Enter initial proportion of non-believers (e.g., 0.3): "))
        
    # Initialize information states w/ default
    initial_states = initialize_states(G, pb0=0.3, pn0=0.3, seed=42)
    
    print(f"Created spatial graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Connection radius: 0.3")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    print(f"\nInitial information distribution:")
    counts = {BELIEVER: 0, NON_BELIEVER: 0, NEUTRAL: 0}
    for state in initial_states.values():
        counts[state] += 1
    print(f"  Believers: {counts[BELIEVER]} ({counts[BELIEVER]/G.number_of_nodes():.1%})")
    print(f"  Non-believers: {counts[NON_BELIEVER]} ({counts[NON_BELIEVER]/G.number_of_nodes():.1%})")
    print(f"  Neutrals: {counts[NEUTRAL]} ({counts[NEUTRAL]/G.number_of_nodes():.1%})")
    
    # Run spatial information spread simulation
    print(f"\nRunning spatial information spread simulation...")
    states_history, global_stats = simulate_network(
        G, initial_states, T=50, 
        pnb=0.1,   # Rate of non-believer → believer conversion
        pbn=0.05,  # Rate of believer → non-believer conversion  
        r=0.5      # Strength of spatial peer pressure
    )
    
    # Visualize spatial information spread
    # Adjust visualization timesteps based on network size
    if len(states_history) > 4:
        timesteps_to_show = [0, len(states_history)//4, len(states_history)//2, len(states_history)-1]
    else:
        timesteps_to_show = list(range(len(states_history)))
    
    visualize_spatial_network_evolution(G, pos, states_history, global_stats, timesteps_to_show,
                                      walker_path, spanning_tree, coverage_stats)
    
    # Analyze final spatial information distribution
    final_stats = global_stats[-1]
    print(f"\nFinal information distribution after spatial spread:")
    print(f"  Believers: {final_stats['pb']:.3f} ({final_stats['pb']:.1%})")
    print(f"  Non-believers: {final_stats['pn']:.3f} ({final_stats['pn']:.1%})")
    print(f"  Neutrals: {final_stats['pneutral']:.3f} ({final_stats['pneutral']:.1%})")
    
    # Calculate change from initial distribution
    initial_stats = global_stats[0]
    print(f"\nSpatial influence effects:")
    print(f"  Believer change: {final_stats['pb'] - initial_stats['pb']:+.3f}")
    print(f"  Non-believer change: {final_stats['pn'] - initial_stats['pn']:+.3f}")
    print(f"  Neutral change: {final_stats['pneutral'] - initial_stats['pneutral']:+.3f}")