import sys
import os
import streamlit as st
import torch
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

# Set page configuration first
st.set_page_config(layout="wide", page_title="GCN Dashboard", page_icon="📊")

# Add the parent directory (src) to Python's module search path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import GCN
from graph_loader import construct_graph
from data_loader import load_elliptic_dataset
from visualization.utils import plot_embeddings, plot_interactive_graph

# --------------------------------------------------
# Load Data and Model (Cached)
# --------------------------------------------------
@st.cache_resource
def load_model_and_data():
    dataset_df, edgelist_df = load_elliptic_dataset()
    data = construct_graph(dataset_df, edgelist_df)
    model = GCN(
        input_dim=data.num_node_features,
        hidden_dim=128,
        output_dim=2,
        dropout=0.5
    )
    model.eval()
    return data, model

data, model = load_model_and_data()

# --------------------------------------------------
# Get Predictions (Ignore model and data for caching by using leading underscore)
# --------------------------------------------------
@st.cache_resource
def get_predictions(_model, _data):
    with torch.no_grad():
        out = _model(_data.x, _data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
    return preds

preds = get_predictions(model, data)

# --------------------------------------------------
# Sidebar: Visualization Settings
# --------------------------------------------------
st.sidebar.header("Visualization Settings")
show_predicted = st.sidebar.checkbox("Show Predicted Labels", value=False)
embedding_method = st.sidebar.selectbox("Embedding Method", ["UMAP", "t-SNE"])
n_components = st.sidebar.selectbox("Dimensions", [2, 3])
perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30) if embedding_method=="t-SNE" else 15

# Choose label colors based on toggle.
node_colors = preds if show_predicted else data.y.cpu().numpy()

# --------------------------------------------------
# Main Dashboard Content
# --------------------------------------------------
st.title("GCN Model Visualization Dashboard")

# 1. Node Embeddings Visualization
st.header("Node Embeddings")
if embedding_method == "t-SNE":
    reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
else:
    reducer = UMAP(n_components=n_components, random_state=42)

with st.spinner("Computing embeddings..."):
    embeddings = model.conv1(data.x, data.edge_index).detach().cpu().numpy()
    if "embeddings_reduced" not in st.session_state:
        st.session_state.embeddings_reduced = reducer.fit_transform(embeddings)

fig = plot_embeddings(
    st.session_state.embeddings_reduced,
    node_colors,
    method=embedding_method,
    n_components=n_components,
    perplexity=perplexity
)
st.plotly_chart(fig, use_container_width=True)

# 2. Interactive Graph Visualization
st.header("Graph Structure")
st.write("Displaying a subgraph with 15,000 nodes (nodes colored by " + ("predicted" if show_predicted else "true") + " labels).")
try:
    import torch_geometric.utils as pyg_utils
    degrees = pyg_utils.degree(data.edge_index[0], data.num_nodes)
    # Select top 15,000 nodes by degree, or if fewer nodes exist, take all.
    num_to_select = min(15000, data.num_nodes)
    top_nodes = degrees.topk(num_to_select).indices.tolist()
    # Convert to tensor
    subgraph_indices = torch.tensor(top_nodes, dtype=torch.long)
    sub_data = data.subgraph(subgraph_indices)
except Exception as e:
    st.write("Error computing top nodes; using random subgraph instead.")
    all_indices = list(range(data.num_nodes))
    selected_nodes = np.random.choice(all_indices, min(15000, data.num_nodes), replace=False).tolist()
    subgraph_indices = torch.tensor(selected_nodes, dtype=torch.long)
    sub_data = data.subgraph(subgraph_indices)

html_file = plot_interactive_graph(sub_data, colors=node_colors)
with open(html_file, "r", encoding="utf-8") as f:
    html_content = f.read()
st.components.v1.html(html_content, height=600, scrolling=True)

# 3. Sidebar Summary Information
st.sidebar.header("Data Summary")
st.sidebar.write(f"Total Nodes: {data.num_nodes}")
unique_true, counts_true = np.unique(data.y.cpu().numpy(), return_counts=True)
unique_pred, counts_pred = np.unique(preds, return_counts=True)
st.sidebar.write(f"True Class Distribution: {dict(zip(unique_true, counts_true))}")
st.sidebar.write(f"Predicted Class Distribution: {dict(zip(unique_pred, counts_pred))}")
