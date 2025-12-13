
import argparse
import networkx as nx
import numpy as np
import os
import sys

# Add parent directory to path to allow imports from paper_experiments
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Attempt to import the utility function
    from paper_experiments.utils_lges import get_random_data, RANDOM_SEED
except ImportError:
    # Provide a fallback implementation if the import fails
    print("Warning: Could not import 'utils_lges'. Using a fallback data generation method.", file=sys.stderr)
    RANDOM_SEED = 42
    def get_random_data(dag_adjacency, n_samples, model, random_state):
        rng = np.random.default_rng(random_state)
        n_variables = dag_adjacency.shape[0]
        data = np.zeros((n_samples, n_variables))
        
        # Get topological sort of nodes
        topological_order = list(nx.topological_sort(nx.DiGraph(dag_adjacency)))

        for i in topological_order:
            parents = np.where(dag_adjacency[:, i] == 1)[0]
            
            if len(parents) == 0:
                # Root node
                data[:, i] = rng.normal(loc=0, scale=1, size=n_samples)
            else:
                # Child node, linear-gaussian model
                coeffs = rng.uniform(low=-1.5, high=1.5, size=len(parents))
                noise = rng.normal(loc=0, scale=0.5, size=n_samples)
                parent_data = data[:, parents]
                data[:, i] = parent_data @ coeffs + noise
        
        return [data]


def create_scale_free_dag(n_nodes, seed):
    """Generates a scale-free Directed Acyclic Graph (DAG)."""
    # Use the Barabasi-Albert model which is known to generate scale-free networks
    g = nx.barabasi_albert_graph(n=n_nodes, m=2, seed=seed)
    
    # Convert to a DiGraph and enforce DAG property by ordering nodes
    dag = nx.DiGraph()
    dag.add_nodes_from(g.nodes())
    
    # Add edges such that they go from a lower index node to a higher index one, ensuring no cycles
    for u, v in g.edges():
        if u < v:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
            
    return dag


def main():
    """
    Generates sample data from a random scale-free DAG and saves it to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Generate sample data for causal discovery algorithms.")
    parser.add_argument('--n-variables', type=int, required=True, help='Number of variables (nodes) in the DAG.')
    parser.add_argument('--n-samples', type=int, required=True, help='Number of data samples to generate.')
    parser.add_argument('--output-file', type=str, required=True, help='Path to save the generated data CSV file.')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    
    print(f"Generating a scale-free DAG with {args.n_variables} variables...")
    dag = create_scale_free_dag(n_nodes=args.n_variables, seed=args.seed)
    dag_adjacency = nx.to_numpy_array(dag, nodelist=sorted(dag.nodes()), weight=None)
    
    print(f"Generating {args.n_samples} samples from a linear-gaussian model...")
    # The imported get_random_data returns a list, so we take the first element
    sample_data = get_random_data(
        dag_adjacency,
        args.n_samples,
        model="linear-gaussian",
        random_state=args.seed
    )[0]
    
    print(f"Saving data to {args.output_file}...")
    np.savetxt(args.output_file, sample_data, delimiter=',')
    
    print("Data generation complete.")

if __name__ == '__main__':
    main()
