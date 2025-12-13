import argparse
import networkx as nx
import numpy as np
import sys

def get_random_data(dag_adjacency, n_samples, random_state):
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
    
    return data

def create_scale_free_dag(n_nodes, m, seed):
    """Generates a scale-free Directed Acyclic Graph (DAG)."""
    g = nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)
    
    # Convert to a DiGraph and enforce DAG property by ordering nodes
    dag = nx.DiGraph()
    dag.add_nodes_from(g.nodes())
    
    # Add edges such that they go from a lower index node to a higher index one
    for u, v in g.edges():
        if u < v:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
            
    return dag

def save_ground_truth(dag, filename):
    nodes = sorted(list(dag.nodes()))
    n = len(nodes)
    adj = nx.to_numpy_array(dag, nodelist=nodes, weight=None)
    
    with open(filename, 'w') as f:
        # Write nodes header
        f.write(','.join(map(str, nodes)) + '\n')
        
        # Write adjacency matrix
        for i in range(n):
            row = []
            for j in range(n):
                if adj[i, j] != 0:
                    row.append('1')
                else:
                    row.append('0')
            f.write(','.join(row) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate sample data and ground truth")
    parser.add_argument('--n-variables', type=int, required=True)
    parser.add_argument('--n-samples', type=int, required=True)
    parser.add_argument('--avg-degree', type=float, default=2.0)
    parser.add_argument('--output-data', type=str, required=True)
    parser.add_argument('--output-truth', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    m = int(round(args.avg_degree / 2))
    if m < 1: m = 1
    
    print(f"Generating DAG with {args.n_variables} variables, m={m}...")
    dag = create_scale_free_dag(n_nodes=args.n_variables, m=m, seed=args.seed)
    dag_adjacency = nx.to_numpy_array(dag, nodelist=sorted(dag.nodes()), weight=None)
    
    print(f"Saving ground truth to {args.output_truth}...")
    save_ground_truth(dag, args.output_truth)
    
    print(f"Generating {args.n_samples} samples...")
    sample_data = get_random_data(
        dag_adjacency,
        args.n_samples,
        random_state=args.seed
    )
    
    print(f"Saving data to {args.output_data}...")
    np.save(args.output_data, sample_data)
    print("Done.")

if __name__ == '__main__':
    main()
