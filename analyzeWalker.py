import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
from collections import defaultdict
import seaborn as sns

# Import all functions from your main file
# Assuming your main file is named 'randomGraph.py'
from randomGraph import *

def run_single_trial(n_nodes, radius=0.3, seed=None):
    """
    Run a single trial of walker efficiency analysis.
    Returns dictionary with all metrics or None if failed.
    """
    try:
        # Create spatial graph
        G, pos = create_spatial_graph(n_nodes=n_nodes, radius=radius, seed=seed)
        
        # Check if we have enough edges for meaningful analysis
        if G.number_of_edges() == 0:
            return None
            
        # Generate spanning tree and perform random walk
        spanning_tree = generate_random_spanning_tree(G, seed=seed)
        
        # Skip if graph isn't connected
        if spanning_tree is None:
            return None
            
        # Perform random walk and collect efficiency metrics
        walker_path, coverage_stats = random_walk_coverage(G, spanning_tree, seed=seed)
        
        # Compile all metrics
        trial_results = {
            'n_nodes': n_nodes,
            'n_edges': G.number_of_edges(),
            'edge_density': G.number_of_edges() / n_nodes if n_nodes > 0 else 0,
            'avg_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G),
            'graph_diameter': nx.diameter(G) if nx.is_connected(G) else None,
            
            # Walker efficiency metrics
            'total_steps': coverage_stats['total_steps'],
            'steps_to_full_coverage': coverage_stats['steps_to_full_coverage'],
            'coverage_ratio': coverage_stats['coverage_ratio'],
            'efficiency_ratio': coverage_stats['efficiency_ratio'],
            'total_revisits': coverage_stats['total_revisits'],
            'backtracking_ratio': coverage_stats['backtracking_ratio'],
            
            # Derived metrics
            'avg_visiting_time_per_node': coverage_stats['steps_to_full_coverage'] / n_nodes if n_nodes > 0 else 0,
            'steps_per_edge': coverage_stats['total_steps'] / G.number_of_edges() if G.number_of_edges() > 0 else 0,
            
            # Include spatial metrics if available
            'total_spatial_distance': coverage_stats.get('total_spatial_distance', None),
            'average_step_distance': coverage_stats.get('average_step_distance', None),
            'spatial_efficiency': coverage_stats.get('spatial_efficiency', None),
            'mean_coverage_time': coverage_stats.get('mean_coverage_time', None),
            'std_coverage_time': coverage_stats.get('std_coverage_time', None),
            'max_coverage_time': coverage_stats.get('max_coverage_time', None),
            
            'success': True
        }
        
        return trial_results
        
    except Exception as e:
        print(f"Trial failed for n_nodes={n_nodes}, seed={seed}: {str(e)}")
        return {
            'n_nodes': n_nodes,
            'success': False,
            'error': str(e)
        }

def run_walker_efficiency_analysis(node_counts=[10, 20, 30, 50, 75, 100], 
                                 n_trials=20, 
                                 radius=0.3,
                                 base_seed=42):
    """
    Run comprehensive walker efficiency analysis across different network sizes.
    
    Args:
        node_counts: List of node counts to test
        n_trials: Number of trials per node count
        radius: Connection radius for spatial graphs
        base_seed: Base seed for reproducibility
    
    Returns:
        Nested dictionary: {node_count: {trial_number: {metrics...}}}
    """
    print("=== Random Walker Efficiency Analysis ===")
    print(f"Testing node counts: {node_counts}")
    print(f"Trials per node count: {n_trials}")
    print(f"Connection radius: {radius}")
    print("=" * 50)
    
    # Initialize results storage
    results = {}
    total_trials = len(node_counts) * n_trials
    completed_trials = 0
    start_time = time.time()
    
    for n_nodes in node_counts:
        print(f"\nTesting {n_nodes} nodes...")
        results[n_nodes] = {}
        successful_trials = 0
        
        for trial in range(n_trials):
            # Use different seed for each trial
            trial_seed = base_seed + completed_trials
            
            # Run single trial
            trial_result = run_single_trial(n_nodes, radius=radius, seed=trial_seed)
            
            # Store result
            results[n_nodes][trial] = trial_result
            
            if trial_result and trial_result.get('success', False):
                successful_trials += 1
            
            completed_trials += 1
            
            # Progress update
            if (trial + 1) % 5 == 0 or trial == n_trials - 1:
                elapsed = time.time() - start_time
                estimated_total = elapsed * total_trials / completed_trials
                remaining = estimated_total - elapsed
                
                print(f"  Trial {trial + 1}/{n_trials} | "
                      f"Success rate: {successful_trials}/{trial + 1} | "
                      f"Time remaining: {remaining/60:.1f}m")
    
    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time/60:.2f} minutes")
    print(f"Total trials: {completed_trials}")
    
    return results

def calculate_summary_statistics(results):
    """
    Calculate summary statistics from trial results.
    
    Returns:
        Dictionary with summary stats for each node count
    """
    summary = {}
    
    for n_nodes, trials in results.items():
        # Extract successful trials only
        successful_trials = [trial for trial in trials.values() 
                           if trial and trial.get('success', False)]
        
        if not successful_trials:
            summary[n_nodes] = {'success_rate': 0, 'n_successful': 0}
            continue
        
        # Calculate statistics for each metric
        metrics = {}
        for key in successful_trials[0].keys():
            if key in ['success', 'error', 'n_nodes']:
                continue
            
            values = [trial[key] for trial in successful_trials 
                     if trial.get(key) is not None]
            
            if values and all(isinstance(v, (int, float)) for v in values):
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
        
        summary[n_nodes] = {
            'success_rate': len(successful_trials) / len(trials),
            'n_successful': len(successful_trials),
            'n_total': len(trials),
            'metrics': metrics
        }
    
    return summary

def visualize_walker_efficiency(summary_stats, save_plots=True):
    """
    Create comprehensive visualizations of walker efficiency analysis.
    """
    # Extract data for plotting
    node_counts = sorted([n for n in summary_stats.keys() 
                         if summary_stats[n]['success_rate'] > 0])
    
    if not node_counts:
        print("No successful trials to visualize!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Random Walker Efficiency Analysis', fontsize=13, fontweight='bold')
    
    # 1. Coverage Time vs. Network Size
    ax = axes[0, 0]
    coverage_times = [summary_stats[n]['metrics']['steps_to_full_coverage']['mean'] 
                     for n in node_counts]
    coverage_stds = [summary_stats[n]['metrics']['steps_to_full_coverage']['std'] 
                    for n in node_counts]
    
    ax.errorbar(node_counts, coverage_times, yerr=coverage_stds, 
                marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Nodes (N)')
    ax.set_ylabel('Steps to Full Coverage')
    ax.set_title('Coverage Time Scaling')
    ax.grid(True, alpha=0.3)
    
    # 2. Average Visiting Time per Node
    ax = axes[0, 1]
    avg_times = [summary_stats[n]['metrics']['avg_visiting_time_per_node']['mean'] 
                for n in node_counts]
    avg_stds = [summary_stats[n]['metrics']['avg_visiting_time_per_node']['std'] 
               for n in node_counts]
    
    ax.errorbar(node_counts, avg_times, yerr=avg_stds, 
                marker='s', capsize=5, linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Number of Nodes (N)')
    ax.set_ylabel('Average Visiting Time per Node')
    ax.set_title('Per-Node Visiting Time')
    ax.grid(True, alpha=0.3)
    
    # 3. Efficiency Ratio vs. Network Size
    ax = axes[0, 2]
    efficiency_ratios = [summary_stats[n]['metrics']['efficiency_ratio']['mean'] 
                        for n in node_counts]
    efficiency_stds = [summary_stats[n]['metrics']['efficiency_ratio']['std'] 
                      for n in node_counts]
    
    ax.errorbar(node_counts, efficiency_ratios, yerr=efficiency_stds, 
                marker='^', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Nodes (N)')
    ax.set_ylabel('Efficiency Ratio')
    ax.set_title('Walker Efficiency vs. Network Size')
    ax.grid(True, alpha=0.3)
    
    # 4. Edge Density vs. Coverage Time
    ax = axes[1, 0]
    edge_densities = [summary_stats[n]['metrics']['edge_density']['mean'] 
                     for n in node_counts]
    
    ax.scatter(edge_densities, coverage_times, s=100, alpha=0.7, c=node_counts, 
               cmap='viridis')
    ax.set_xlabel('Edge Density (C/N)')
    ax.set_ylabel('Steps to Full Coverage')
    ax.set_title('Coverage Time vs. Edge Density')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Number of Nodes')
    
    # 5. Backtracking Ratio
    ax = axes[1, 1]
    backtrack_ratios = [summary_stats[n]['metrics']['backtracking_ratio']['mean'] 
                       for n in node_counts]
    backtrack_stds = [summary_stats[n]['metrics']['backtracking_ratio']['std'] 
                     for n in node_counts]
    
    ax.errorbar(node_counts, backtrack_ratios, yerr=backtrack_stds, 
                marker='d', capsize=5, linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Nodes (N)')
    ax.set_ylabel('Backtracking Ratio')
    ax.set_title('Backtracking vs. Network Size')
    ax.grid(True, alpha=0.3)
    
    # 6. Log-log plot for scaling analysis
    ax = axes[1, 2]
    ax.loglog(node_counts, coverage_times, 'o-', linewidth=2, markersize=8, 
              label='Observed')
    
    # Fit and plot theoretical scaling curves
    log_n = np.log(node_counts)
    log_coverage = np.log(coverage_times)
    
    # Linear fit in log space
    coeffs = np.polyfit(log_n, log_coverage, 1)
    scaling_exponent = coeffs[0]
    
    # Plot fitted line
    fitted_coverage = np.exp(coeffs[1]) * np.array(node_counts) ** scaling_exponent
    ax.loglog(node_counts, fitted_coverage, '--', linewidth=2, 
              label=f'Power law: N^{scaling_exponent:.2f}')
    
    ax.set_xlabel('Number of Nodes (N)')
    ax.set_ylabel('Steps to Full Coverage')
    ax.set_title('Scaling Analysis (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(top=0.93)  # Leave room for main title
    
    if save_plots:
        plt.savefig('walker_efficiency_analysis.png', dpi=300, bbox_inches='tight')
        print("Plots saved as 'walker_efficiency_analysis.png'")
    
    plt.show()
    
    return scaling_exponent

def save_results(results, summary_stats, filename_prefix='walker_analysis'):
    """
    Save results to files for later analysis.
    """
    # Save raw results as JSON
    with open(f'{filename_prefix}_raw_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary statistics as JSON
    with open(f'{filename_prefix}_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create CSV with summary data
    csv_data = []
    for n_nodes, stats in summary_stats.items():
        if stats['success_rate'] == 0:
            continue
            
        row = {
            'n_nodes': n_nodes,
            'success_rate': stats['success_rate'],
            'n_successful': stats['n_successful']
        }
        
        # Add mean values for key metrics
        for metric_name, metric_stats in stats['metrics'].items():
            row[f'{metric_name}_mean'] = metric_stats['mean']
            row[f'{metric_name}_std'] = metric_stats['std']
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(f'{filename_prefix}_summary.csv', index=False)
    
    print(f"Results saved:")
    print(f"  - Raw data: {filename_prefix}_raw_results.json")
    print(f"  - Summary: {filename_prefix}_summary.json")
    print(f"  - CSV: {filename_prefix}_summary.csv")

def print_analysis_report(summary_stats, scaling_exponent):
    """
    Print a comprehensive analysis report.
    """
    print("\n" + "="*60)
    print("RANDOM WALKER EFFICIENCY ANALYSIS REPORT")
    print("="*60)
    
    for n_nodes in sorted(summary_stats.keys()):
        stats = summary_stats[n_nodes]
        
        if stats['success_rate'] == 0:
            print(f"\nN={n_nodes}: No successful trials")
            continue
        
        print(f"\nN={n_nodes} nodes (Success rate: {stats['success_rate']:.1%}):")
        print(f"  Average visiting time per node: {stats['metrics']['avg_visiting_time_per_node']['mean']:.2f} ± {stats['metrics']['avg_visiting_time_per_node']['std']:.2f}")
        print(f"  Steps to full coverage: {stats['metrics']['steps_to_full_coverage']['mean']:.1f} ± {stats['metrics']['steps_to_full_coverage']['std']:.1f}")
        print(f"  Efficiency ratio: {stats['metrics']['efficiency_ratio']['mean']:.3f} ± {stats['metrics']['efficiency_ratio']['std']:.3f}")
        print(f"  Edge density (C/N): {stats['metrics']['edge_density']['mean']:.2f} ± {stats['metrics']['edge_density']['std']:.2f}")
        print(f"  Backtracking ratio: {stats['metrics']['backtracking_ratio']['mean']:.3f} ± {stats['metrics']['backtracking_ratio']['std']:.3f}")
    
    print(f"\nSCALING ANALYSIS:")
    print(f"  Coverage time scales as N^{scaling_exponent:.2f}")
    
    if scaling_exponent < 1.5:
        scaling_desc = "Sub-linear (very efficient)"
    elif scaling_exponent < 2.0:
        scaling_desc = "Linear to sub-quadratic (efficient)"
    elif scaling_exponent < 2.5:
        scaling_desc = "Quadratic (moderate efficiency)"
    else:
        scaling_desc = "Super-quadratic (poor scaling)"
    
    print(f"  Scaling interpretation: {scaling_desc}")
    print("="*60)

def main():
    """
    Main analysis function - run the complete walker efficiency study.
    """
    # Configuration
    node_counts = [10, 20, 30, 50, 75, 100]  # Adjust based on your computational resources
    n_trials = 10
    radius = 0.3
    
    print("Starting Random Walker Efficiency Analysis...")
    print(f"This will run {len(node_counts) * n_trials} total simulations.")
    print("This may take several minutes to complete.\n")
    
    # Run the analysis
    results = run_walker_efficiency_analysis(
        node_counts=node_counts,
        n_trials=n_trials,
        radius=radius
    )
    
    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(results)
    
    # Create visualizations and get scaling exponent
    scaling_exponent = visualize_walker_efficiency(summary_stats)
    
    # Print comprehensive report
    print_analysis_report(summary_stats, scaling_exponent)
    
    # Save results for future analysis
    save_results(results, summary_stats)
    
    return results, summary_stats

if __name__ == "__main__":
    results, summary_stats = main()