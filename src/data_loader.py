import pandas as pd
import os

def load_elliptic_dataset():
    # Check for required files
    required_files = [
        'data/elliptic/elliptic_txs_features.csv',
        'data/elliptic/elliptic_txs_classes.csv',
        'data/elliptic/elliptic_txs_edgelist.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}. Download from Kaggle and place in /data/elliptic/")

    # Load features with no header and assign column names.
    features = pd.read_csv('data/elliptic/elliptic_txs_features.csv', header=None)
    features.columns = ['txId', 'time'] + [f'feature_{i}' for i in range(1,94)] + [f'aggr_feature_{i}' for i in range(1,73)]

    # Load classes and filter out unknown nodes (class = -1)
    classes = pd.read_csv('data/elliptic/elliptic_txs_classes.csv')
    classes['class'] = classes['class'].map({'unknown': -1, '1': 1, '2': 0})
    classes = classes[classes['class'] != -1]  # Keep only nodes with class 0 or 1

    # Merge features and classes
    dataset_df = pd.merge(features, classes, on='txId')
    dataset_df = dataset_df.reset_index(drop=True)
    
    # Define valid nodes as those in the merged dataset
    valid_nodes = set(dataset_df['txId'].unique())

    # Load edges and rename columns
    edgelist_df = pd.read_csv('data/elliptic/elliptic_txs_edgelist.csv')
    if 'txId1' not in edgelist_df.columns or 'txId2' not in edgelist_df.columns:
        raise ValueError("Edge list must contain 'txId1' and 'txId2' columns.")
    # Rename to 'source' and 'target'
    edgelist_df = edgelist_df.rename(columns={'txId1': 'source', 'txId2': 'target'})
    
    # Filter the edge list so that both endpoints are in valid_nodes.
    edgelist_df = edgelist_df[edgelist_df['source'].isin(valid_nodes) & edgelist_df['target'].isin(valid_nodes)]
    
    # Re-index nodes using only the valid nodes.
    all_nodes = sorted(valid_nodes)
    node_id_to_idx = {node_id: idx for idx, node_id in enumerate(all_nodes)}
    
    # Map the edge list columns to 0-based indices.
    edgelist_df['source'] = edgelist_df['source'].map(node_id_to_idx)
    edgelist_df['target'] = edgelist_df['target'].map(node_id_to_idx)
    
    # Add a column to dataset_df with the new index.
    dataset_df['node_index'] = dataset_df['txId'].map(node_id_to_idx)
    
    return dataset_df, edgelist_df

if __name__ == "__main__":
    dataset_df, edgelist_df = load_elliptic_dataset()
    print("Dataset DF shape:", dataset_df.shape)
    print("Edge List DF shape:", edgelist_df.shape)
