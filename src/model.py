import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        # Dense layer
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = torch.nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = dropout

        # NEW: Track activations for visualization
        self.activations = {}

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        self.activations['conv1'] = x.detach().cpu().numpy()  # NEW
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        self.activations['conv2'] = x.detach().cpu().numpy()  # NEW
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        self.activations['conv3'] = x.detach().cpu().numpy()  # NEW
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Dense layers
        x = self.fc(x)
        x = F.relu(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)  # Predictions
