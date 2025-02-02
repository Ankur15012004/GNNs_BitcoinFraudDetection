import torch
from torch_geometric.data import Data
from data_loader import load_elliptic_dataset

def construct_graph(dataset_df, edgelist_df):
    """
    Constructs a PyTorch Geometric Data object with proper node indexing.
    """
    # Extract node features (exclude metadata columns)
    feature_columns = [col for col in dataset_df.columns if col not in ['txId', 'class', 'node_index', 'time']]
    x = torch.tensor(dataset_df[feature_columns].values, dtype=torch.float)

    # Extract node labels
    y = torch.tensor(dataset_df['class'].values, dtype=torch.long)

    # Extract edges
    edge_index = torch.tensor(edgelist_df.values.T, dtype=torch.long)

    # Extract time information
    time = torch.tensor(dataset_df['time'].values, dtype=torch.float)

    # Verify labels are valid
    assert (y == 0).logical_or(y == 1).all(), "Labels contain invalid values (use data_loader to filter class=-1)"

    # Verify node indices
    max_node_idx = edge_index.max().item()
    assert max_node_idx < len(dataset_df), f"Edge index contains invalid node indices: {max_node_idx} >= {len(dataset_df)}"

    # Create Data object with time attribute
    data = Data(x=x, y=y, edge_index=edge_index, time=time)
    
    print("\nGraph Summary:")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_node_features}, Classes: {data.y.unique().tolist()}")
    
    return data

if __name__ == "__main__":
    dataset_df, edgelist_df = load_elliptic_dataset()
    graph_data = construct_graph(dataset_df, edgelist_df)
