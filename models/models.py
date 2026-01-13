import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from mamba_ssm import Mamba

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__() # Define the model layers
        self.fc1 = nn.Linear(d_model, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True) 
        self.fc3 = nn.Linear(256, d_model, bias=True) 
        self.dropout = nn.Dropout(0.25) 


    def forward(self, x): # Define the forward pass sequence
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x))) 
        x = F.relu(self.fc3(x))
      
        return x
# Mixture of Experts (MoE) Layer Implementation
class MoE(nn.Module):
    def __init__(self, d_model, num_experts, k=1, exploration_weight=0.02):
        """
        Mixture of Experts layer with Top-K Gating and Load Balancing Regularization.
        Args:
            d_model: Input dimension
            num_experts: Number of experts
            k: Number of top experts to activate
            exploration_weight: Weight for load balancing regularization
        """
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([MLP(d_model) for _ in range(num_experts)])
        self.gating = nn.Linear(d_model, num_experts)  # Gating network to select experts
        self.num_experts = num_experts
        self.k = k  # Number of top experts to activate
        self.exploration_weight = exploration_weight  # Regularization weight

    def forward(self, x):
        # Compute gating weights
        gates = torch.softmax(self.gating(x), dim=-1)  # Shape: (..., num_experts)

        # Select Top-K experts
        _, top_k_indices = torch.topk(gates, self.k, dim=-1)  # Top-K expert indices
        top_k_gates = torch.zeros_like(gates).scatter_(-1, top_k_indices, gates.gather(-1, top_k_indices))  # Mask for Top-K gates

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # Shape: (..., d_model, num_experts)

        # Combine expert outputs based on Top-K gating weights
        output = torch.einsum('...nde,...ne->...nd', expert_outputs, top_k_gates)  # Weighted sum with Top-K gates

        # Add exploration loss for load balancing
        avg_gates = gates.mean(dim=0)  # Average gating weights across batch
        load_balancing_loss = (avg_gates * torch.log(avg_gates + 1e-6)).sum() / self.num_experts  # Entropy-based loss
        self.load_balancing_loss = -self.exploration_weight * load_balancing_loss  # Minimize imbalance

        return output
    
class SparseMambaNet(nn.Module):
    def __init__(self, args):
        """
        Args:
            input_dim: Dimension of input features
            d_model: Model dimension for Mamba2
            d_state: SSM state expansion factor for Mamba2
            d_conv: Local convolution width for Mamba2
            expand: Block expansion factor for Mamba2
            n_layers: Number of Mamba2 layers to stack
            num_experts: Number of experts in MoE
            num_classes: Number of output classes
        """
        super(SparseMambaNet, self).__init__()
         
        self.input_dim = args.input_dim
        self.d_model = args.d_model
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.n_layers = args.n_layers
        self.num_experts = args.num_experts
        self.num_classes = args.num_classes
        
        # Initial projection layer to project input to d_model
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Layer Normalization as the first layer
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Stack n Mamba layers
        self.mamba_layers = nn.ModuleList([
            BiMambaEncoder(d_model = self.d_model, n_state = self.d_state)
            for _ in range(self.n_layers)
        ])
        
        # Mixture of Experts (MoE) Layer
        self.moe = MoE(self.d_model, self.num_experts)
        
        # Final Classification Layer
        self.fc = nn.Linear(self.d_model, self.num_classes)
        
    def forward(self, x):
        # Project input to d_model
        x = self.input_projection(x)
        
        for layer in self.mamba_layers:
            # Pass through stacked Mamba2 layers
            x = layer(x)
            # Pass through MoE layer    
            x = self.moe(x)
        
        # Final classification
        x = self.fc(x)
        return x


