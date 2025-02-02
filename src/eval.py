import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    accuracy_score
)
from model import GCN
from gat import GAT
from graph_loader import construct_graph
from data_loader import load_elliptic_dataset

def load_model(model_type, checkpoint_path, input_dim, hidden_dim, output_dim, device):
    """Load a trained GCN or GAT model."""
    if model_type == "GCN":
        model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=0.5)
    elif model_type == "GAT":
        model = GAT(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, heads=8, dropout=0.6)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights and move to device
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, data, device):
    """Evaluate the model on the test set and compute metrics."""
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
        probs = torch.exp(out).cpu().numpy()[:, 1]  # Probability of class 1
    
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = preds[data.test_mask.cpu().numpy()]
    y_probs = probs[data.test_mask.cpu().numpy()]

    metrics = {
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    # Compute ROC curve and AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = (fpr, tpr)
    except ValueError:
        metrics['roc_auc'] = None
        metrics['roc_curve'] = None

    return metrics

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Licit', 'Illicit'], rotation=45)
    plt.yticks(tick_marks, ['Licit', 'Illicit'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    """Plot ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()

def main(model_type, checkpoint_path):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset and construct graph
    dataset_df, edgelist_df = load_elliptic_dataset()
    data = construct_graph(dataset_df, edgelist_df)

    # Load the trained model
    input_dim = data.num_node_features
    hidden_dim = 128 if model_type == "GAT" else 256
    output_dim = 2
    
    print(f"\nLoading {model_type} from {checkpoint_path}...")
    
    model = load_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device
    )
    
    # Evaluate the model
    print("\nEvaluating the model...")
    
    metrics = evaluate_model(model=model, data=data.to(device), device=device)

    # Print evaluation results
    print(f"\n=== {model_type} Evaluation ===")
    
