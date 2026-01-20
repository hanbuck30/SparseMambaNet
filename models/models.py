import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from mamba_ssm import Mamba

class SparseMambaNet(nn.Module):
    """
    SparseMambaNet: A hybrid architecture combining Bi-Directional Mamba 
    for temporal modeling and Mixture of Experts (MoE) for efficient feature extraction.
    """
    def __init__(self, args):
        """
        Initialize the SparseMambaNet model.

        Args:
            args: Argument parser object containing model hyperparameters.
                - input_dim: Dimension of input features (e.g., number of EEG channels).
                - d_model: Hidden dimension size for the Mamba model.
                - d_state: SSM state expansion factor.
                - num_experts: Number of experts in the MoE layer.
                - num_classes: Number of output classes for classification.
                - n_layers: Number of Mamba-MoE blocks to stack.
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
        
        # 1. Input Projection
        # LINEAR: Projects the raw input features to the model's hidden dimension (d_model).
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # Layer Normalization applied immediately after projection (Pre-Norm architecture).
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # 2. Stacked Mamba-MoE Layers
        # Constructs a sequence of blocks where each block contains a Bi-Mamba encoder and an MoE module.
        self.mamba_layers = nn.ModuleList([
            SparseMambaMoEBlock(
                d_model=self.d_model, 
                n_state=self.d_state, 
                num_experts=self.num_experts, 
                k=1  # Top-1 expert activation for sparsity
            ) 
            for _ in range(self.n_layers)
        ])
        
        # Note: The standalone self.moe here is redundant if MoE is inside the block, 
        # but kept for consistency if needed for auxiliary purposes.
        # self.moe = MoE(self.d_model, self.num_experts)
        
        # 3. Final Classification Head
        # Projects the processed latent representation to the output class probabilities.
        self.fc = nn.Linear(self.d_model, self.num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (Batch, Time, Input_Dim) or (Batch, Input_Dim, Time) depending on preprocessing.
        """
        # Linear Projection to d_model
        x = self.input_projection(x)
        
        # (Optional) Apply Layer Normalization
        # x = self.layer_norm(x)
        
        # Pass input through the stacked SparseMambaMoE Blocks
        for layer in self.mamba_layers:
            x = layer(x)
        
        # Final Classification
        x = self.fc(x)
        return x

class SparseMambaMoEBlock(nn.Module):
    """
    A unified block containing Bi-Directional Mamba for temporal mixing 
    and Mixture of Experts (MoE) for channel mixing (replacing standard MLP).
    """
    def __init__(self, d_model, n_state, num_experts, k):
        super().__init__()
        
        # Normalization layer before Mamba
        self.norm1 = nn.LayerNorm(d_model)
        
        # Bi-Mamba: Utilizes the Mamba SSM for sequence modeling.
        # Note: We share the same Mamba module or use distinct ones depending on implementation.
        # Here, we reuse the module instance but apply it bi-directionally.
        self.mamba = Mamba(d_model, n_state) 
        
        # Normalization layer before MoE
        self.norm2 = nn.LayerNorm(d_model)
        
        # Mixture of Experts (MoE)
        # Replaces the traditional dense Feed-Forward Network (FFN) to capture diverse feature patterns.
        self.moe = MoE(d_model, num_experts, k=k)

    def forward(self, x):
        # --- Path 1: Bi-Directional Mamba (Temporal Mixing) ---
        residual = x
        x_norm = self.norm1(x)
        
        # Forward Mamba: Process sequence from t=0 to T
        x_fwd = self.mamba(x_norm)
        
        # Backward Mamba: Flip sequence, process, then flip back
        # This allows the model to capture non-causal dependencies (future context).
        x_bwd = torch.flip(self.mamba(torch.flip(x_norm, [1])), [1])
        
        # Combine forward and backward outputs
        x_mamba = x_fwd + x_bwd
        
        # Residual connection
        x = residual + x_mamba
        
        # --- Path 2: Mixture of Experts (Channel Mixing) ---
        residual = x
        x_norm = self.norm2(x)
        
        # Pass through the sparse expert layers
        x_moe = self.moe(x_norm)
        
        # Residual connection
        return residual + x_moe
        
# Mixture of Experts (MoE) Layer Implementation
class MoE(nn.Module):
    def __init__(self, d_model, num_experts, k=1, exploration_weight=0.02):
        """
        Sparsely Activated Mixture of Experts Layer.
        
        Args:
            d_model: Dimension of the input and output.
            num_experts: Total number of experts available.
            k: Number of experts to activate for each token (Top-K).
            exploration_weight: Coefficient for the load balancing loss.
        """
        super(MoE, self).__init__()
        
        # Define a list of Expert networks (Simple MLPs)
        self.experts = nn.ModuleList([MLP(d_model) for _ in range(num_experts)])
        
        # Gating Network: A linear layer to predict the "routing" probability for each expert
        self.gating = nn.Linear(d_model, num_experts)
        
        self.num_experts = num_experts
        self.k = k
        self.exploration_weight = exploration_weight
        self.load_balancing_loss = 0.0 # Stores auxiliary loss

    def forward(self, x):
        # 1. Gating Mechanism
        # Compute probabilities for each expert: (Batch, Seq_Len, Num_Experts)
        gates = torch.softmax(self.gating(x), dim=-1) 

        # 2. Top-K Selection
        # Find indices and values of the top-k highest scoring experts
        # top_k_gates: Weights for the selected experts
        # top_k_indices: Indices of the selected experts
        _, top_k_indices = torch.topk(gates, self.k, dim=-1) 
        
        # Create a sparse mask for the selected experts
        # scatter_(-1, indices, values): Places the top-k values back into a zero-filled tensor
        top_k_gates = torch.zeros_like(gates).scatter_(-1, top_k_indices, gates.gather(-1, top_k_indices)) 

        # 3. Expert Computation
        # Compute output of ALL experts (Note: In optimized CUDA implementations, this would be sparse)
        # expert_outputs shape: (Batch, Seq_Len, d_model, Num_Experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1) 

        # 4. Weighted Aggregation
        # Sum the expert outputs weighted by the gating score (Top-K only)
        # Einstein summation: Combine last dimension (experts)
        output = torch.einsum('...nde,...ne->...nd', expert_outputs, top_k_gates) 

        # 5. Load Balancing Loss (Auxiliary Loss)
        # Encourages the router to use all experts equally over the batch.
        avg_gates = gates.mean(dim=0) # Average usage of each expert in the batch
        
        # Entropy-based loss: Minimizes the variance of expert usage
        load_balancing_loss = (avg_gates * torch.log(avg_gates + 1e-6)).sum() / self.num_experts 
        self.load_balancing_loss = -self.exploration_weight * load_balancing_loss 

        return output
