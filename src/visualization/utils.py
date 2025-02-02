import numpy as np
from pyvis.network import Network
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP

def plot_embeddings(embeddings, labels, method="t-SNE", n_components=2, perplexity=30):
    """
    Reduces dimensionality of node embeddings using t-SNE or UMAP and returns
    a Plotly scatter plot figure.
    """
    # Convert labels to numpy array if needed
    labels = np.array(labels)

    if method == "t-SNE":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    else:
        reducer = UMAP(n_components=n_components, random_state=42)
    emb_reduced = reducer.fit_transform(embeddings)
    
    if n_components == 2:
        fig = px.scatter(
            x=emb_reduced[:, 0],
            y=emb_reduced[:, 1],
            color=labels.astype(str),
            labels={"color": "Class"},
            title=f"{method} Projection (2D)"
        )
    else:
        fig = px.scatter_3d(
            x=emb_reduced[:, 0],
            y=emb_reduced[:, 1],
            z=emb_reduced[:, 2],
            color=labels.astype(str),
            labels={"color": "Class"},
            title=f"{method} Projection (3D)"
        )
    return fig

def plot_interactive_graph(data, colors, num_nodes=15000):
    """
    Creates an interactive subgraph visualization using PyVis.
    Ensures that the 'colors' input is a NumPy array.
    
    Args:
        data (Data): PyG Data object.
        colors: Predictions or true labels (list or array-like) used for node coloring.
        num_nodes (int): The number of nodes to display in the subgraph.
    
    Returns:
        str: The filename of the generated HTML graph.
    """
    # Convert colors to a NumPy array if not already
    colors = np.array(colors)
    
    # Select a random subset of node indices
    all_indices = list(range(data.num_nodes))
    selected_nodes = np.random.choice(all_indices, min(num_nodes, data.num_nodes), replace=False).tolist()
    
    net = Network(height="600px", width="100%", notebook=False)
    net.barnes_hut()  # enable physics-based layout
    
    # Add nodes with color based on provided colors (predicted or true)
    for i in selected_nodes:
        node_id = int(i)
        # Color the node: red if label==1, else blue
        node_color = "red" if colors[i] == 1 else "blue"
        net.add_node(
            node_id,
            label=f"Node {node_id}",
            title=f"True: {data.y[i].item()}, Pred: {colors[i]}",
            color=node_color
        )
    
    # Add edges (only those connecting nodes in selected_nodes)
    edge_index = data.edge_index.cpu().numpy()
    for src, tgt in edge_index.T:
        src, tgt = int(src), int(tgt)
        if src in selected_nodes and tgt in selected_nodes:
            net.add_edge(src, tgt)
    
    net.save_graph("subgraph.html")
    return "subgraph.html"
