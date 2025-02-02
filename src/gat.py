import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 heads=8, 
                 dropout=0.6,
                 attn_dropout=0.3,  # Separate attention dropout
                 num_layers=3,       # Additional layer
                 concat=True):
        super(GAT, self).__init__()
        
        self.dropout = dropout
        self.concat = concat
        
        # Input attention layer
        self.conv1 = GATConv(
            input_dim, 
            hidden_dim, 
            heads=heads,
            dropout=attn_dropout,
            add_self_loops=False  # Experiment with/without
        )
        self.bn1 = BatchNorm(hidden_dim * heads)
        
        # Intermediate layers with residual connections
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=heads,
            dropout=attn_dropout,
            concat=concat
        )
        self.bn2 = BatchNorm(hidden_dim * (heads if concat else 1))
        
        # Additional layer with dynamic attention heads
        self.conv3 = GATConv(
            hidden_dim * (heads if concat else 1),
            hidden_dim,
            heads=heads//2 if concat else 1,  # Reduce heads gradually
            dropout=attn_dropout,
            concat=concat
        )
        
        # Final projection
        self.fc = nn.Linear(hidden_dim * (heads//2 if concat else 1), output_dim)
        
        # Skip connections
        self.skip = nn.Linear(input_dim, hidden_dim * (heads//2 if concat else 1))
        
        # Attention temperature (learnable parameter)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.skip.weight)

    def forward(self, x, edge_index):
        # Initial projection
        x_initial = self.skip(x)
        
        # Layer 1
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2 with residual
        x = F.elu(self.bn2(self.conv2(x, edge_index)) + x)  # Residual
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3 with attention temperature
        x = self.conv3(x * self.temperature, edge_index)  # Scaled attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final projection
        return F.log_softmax(self.fc(x + x_initial), dim=1)  # Skip connection
