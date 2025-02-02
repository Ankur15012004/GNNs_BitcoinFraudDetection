import math
import torch
import torch.nn.functional as F
from model import GCN
from gat import GAT
from graph_loader import construct_graph
from data_loader import load_elliptic_dataset
from sklearn.metrics import f1_score

def compute_class_weights(data):
    """Computes class weights with logarithmic smoothing."""
    train_labels = data.y[data.train_mask]
    count0 = (train_labels == 0).sum().item()
    count1 = (train_labels == 1).sum().item()
    total = count0 + count1
    weight0 = math.log((total / (count0 + 1e-5)) + 1)
    weight1 = math.log((total / (count1 + 1e-5)) + 1)
    return torch.tensor([weight0, weight1], dtype=torch.float)

def train(model, optimizer, data, class_weights):
    """Training loop for both GCN and GAT."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask],
                     weight=class_weights.to(out.device))
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask_name):
    """Evaluation function for both models."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
    mask = getattr(data, mask_name)
    y_true = data.y[mask].cpu().numpy()  # Move to CPU for sklearn
    y_pred = pred[mask].cpu().numpy()
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'acc': (y_true == y_pred).mean()
    }

def train_and_evaluate(model_class, model_name, data, input_dim, hidden_dim, output_dim, save_path, **kwargs):
    """Train and evaluate a single model architecture."""
    print(f"\n{'='*40}\nTraining {model_name}\n{'='*40}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model and optimizer
    model = model_class(input_dim=input_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, **kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Move data to device
    data = data.to(device)
    
    # Compute class weights
    class_weights = compute_class_weights(data).to(device)
    
    best_val_f1 = 0.0
    for epoch in range(200):
        loss = train(model, optimizer, data, class_weights)
        
        if (epoch+1) % 10 == 0 or epoch == 200:
            val_metrics = evaluate(model, data, 'val_mask')
            print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val F1={val_metrics['f1']:.4f}")
            
            # Save the best model based on validation F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save(model.state_dict(), save_path)
                print(f"Saved new best {model_name} model at epoch {epoch+1}")
    
    # Final evaluation on test set
    test_metrics = evaluate(model, data, 'test_mask')
    print(f"\n{model_name} Final Test Performance:")
    print(f"F1: {test_metrics['f1']:.4f}, Accuracy: {test_metrics['acc']:.4f}")
    
if __name__ == "__main__":
    # Load dataset and construct graph
    dataset_df, edgelist_df = load_elliptic_dataset()
    data = construct_graph(dataset_df, edgelist_df)

    # Create splits (60% train, 20% val, 20% test)
    num_nodes = data.num_nodes
    perm = torch.randperm(num_nodes)
    
    train_size = int(0.6 * num_nodes)
    val_size   = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size:train_size+val_size]] = True
    test_mask[perm[train_size+val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    print(f"Train nodes: {data.train_mask.sum().item()}")
    print(f"Validation nodes: {data.val_mask.sum().item()}")
    print(f"Test nodes: {data.test_mask.sum().item()}")

    input_dim  = data.num_node_features
    hidden_dim = 256
    output_dim = 2

    # Train GCN and save the best model
    train_and_evaluate(
        GCN,
        "GCN",
        data,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=0.5,
        save_path="best_gcn_model.pth"
    )

    # Train GAT and save the best model
    train_and_evaluate(
        GAT,
        "GAT",
        data,
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=output_dim,
        heads=8,
        dropout=0.6,
        save_path="best_gat_model.pth"
    )
