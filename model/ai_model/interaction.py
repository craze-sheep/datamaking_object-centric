"""Force-aware Graph Neural Network for object interaction modeling.

Key design:
  - Each object is a node with features from the encoder
  - Edges are constructed from the force matrix (explicit physics signal)
  - Edge features encode: force vector, relative position, relative velocity, distance
  - Message passing with edge features (not just node features)
  - N <= 7, so dense adjacency is fine (no need for sparse PyG)

Key differences from baseline:
  - Baseline: dense pairwise MLP with raw force + relative state as edge features
  - Ours: explicit GNN with separate message/aggregation/update functions
  - We add physics priors: force direction, distance-based attention, momentum exchange signal
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ForceAwareEdgeNetwork(nn.Module):
    """Compute edge features from node states + force matrix.

    Edge feature for (i, j) includes:
      - Force from j on i: force_matrix[i, j]  (3-dim)
      - Force from i on j: force_matrix[j, i]  (3-dim)
      - Relative position: pos_i - pos_j        (3-dim, extracted from state)
      - Relative velocity: vel_i - vel_j        (3-dim, extracted from state)
      - Distance: ||pos_i - pos_j||             (1-dim)
      - Force magnitude: ||force_ij||           (1-dim)
    Total: 14-dim → projected to edge_hidden_dim
    """

    def __init__(self, state_dim: int, edge_dim: int):
        super().__init__()
        # Edge input: force_ij(3) + force_ji(3) + rel_pos(3) + rel_vel(3) + dist(1) + force_mag(1) = 14
        edge_input_dim = 14
        self.edge_net = nn.Sequential(
            nn.Linear(edge_input_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(
        self,
        dyn_state: torch.Tensor,     # [B, T, N, state_dim] (raw, for pos/vel extraction)
        force_matrix: torch.Tensor,   # [B, T, N, N, 3]
        valid_mask: torch.Tensor,     # [B, N]
    ) -> torch.Tensor:
        """
        Returns:
            edge_feat: [B, T, N, N, edge_dim]
        """
        B, T, N, _ = dyn_state.shape

        # Extract position (idx 0:3) and velocity (idx 7:10) from state
        pos = dyn_state[..., 0:3]    # [B, T, N, 3]
        vel = dyn_state[..., 7:10]   # [B, T, N, 3]

        # Relative position and velocity
        # pos_i - pos_j: [B, T, N, 1, 3] - [B, T, 1, N, 3]
        rel_pos = pos.unsqueeze(3) - pos.unsqueeze(2)  # [B, T, N, N, 3]
        rel_vel = vel.unsqueeze(3) - vel.unsqueeze(2)  # [B, T, N, N, 3]

        # Distance
        dist = rel_pos.norm(dim=-1, keepdim=True)  # [B, T, N, N, 1]

        # Force magnitude
        force_mag = force_matrix.norm(dim=-1, keepdim=True)  # [B, T, N, N, 1]

        # Concatenate edge features
        edge_input = torch.cat([
            force_matrix,      # [B, T, N, N, 3] - force from j on i
            force_matrix.transpose(2, 3),  # [B, T, N, N, 3] - force from i on j
            rel_pos,           # [B, T, N, N, 3]
            rel_vel,           # [B, T, N, N, 3]
            dist,              # [B, T, N, N, 1]
            force_mag,         # [B, T, N, N, 1]
        ], dim=-1)  # [B, T, N, N, 14]

        # Project
        edge_feat = self.edge_net(edge_input)  # [B, T, N, N, edge_dim]

        # Mask out invalid edges
        # valid_mask: [B, N] -> [B, 1, N, N] pair mask
        pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, N, N]
        pair_mask = pair_mask.unsqueeze(1).float()  # [B, 1, N, N]
        edge_feat = edge_feat * pair_mask.unsqueeze(-1)

        return edge_feat


class GNNSingleLayer(nn.Module):
    """Single GNN message passing layer.

    Message: MLP(node_i, node_j, edge_ij)
    Aggregation: mean over valid neighbors
    Update: GRU-style gate on (old_state, aggregated_messages)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        # Message function: combines sender, receiver, and edge features
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )
        self.attn_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ) if use_attention else None
        # Update function: gated update
        self.update_fn = nn.GRUCell(node_dim, node_dim)

    def forward(
        self,
        node_feat: torch.Tensor,  # [B, T, N, D]
        edge_feat: torch.Tensor,  # [B, T, N, N, E]
        valid_mask: torch.Tensor, # [B, N]
    ) -> torch.Tensor:
        """
        Returns:
            updated_nodes: [B, T, N, D]
        """
        B, T, N, D = node_feat.shape

        # Prepare pairwise node features for message computation
        # node_i: [B, T, N, 1, D], node_j: [B, T, 1, N, D]
        node_i = node_feat.unsqueeze(3).expand(-1, -1, -1, N, -1)
        node_j = node_feat.unsqueeze(2).expand(-1, -1, N, -1, -1)

        # Message input: [B, T, N, N, 2D+E]
        msg_input = torch.cat([node_i, node_j, edge_feat], dim=-1)
        messages = self.message_fn(msg_input)  # [B, T, N, N, D]

        # Mask: only aggregate from valid neighbors, excluding self loops.
        pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)  # [B, N, N]
        eye = torch.eye(N, device=node_feat.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask.bool() & ~eye
        pair_mask = pair_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, N, N, 1]
        messages = messages * pair_mask

        if self.use_attention:
            # GAT-style masked attention over senders. This keeps dense N<=7
            # computation but lets collisions/near contacts dominate messages.
            attn_logits = self.attn_fn(msg_input)
            attn_logits = F.leaky_relu(attn_logits, negative_slope=0.2)
            attn_logits = attn_logits.masked_fill(pair_mask <= 0, -1e4)
            attn = F.softmax(attn_logits, dim=3)
            attn = attn * pair_mask
            aggregated = (messages * attn).sum(dim=3)
        else:
            # Aggregate: mean over neighbors (dim=3 is the sender dimension)
            n_valid = pair_mask.sum(dim=3).clamp(min=1)  # [B, 1, N, 1]
            aggregated = messages.sum(dim=3) / n_valid  # [B, T, N, D]

        # Update: GRU-style gated update
        # Reshape for GRUCell: [B*T*N, D]
        old_flat = node_feat.reshape(B * T * N, D)
        agg_flat = aggregated.reshape(B * T * N, D)
        updated_flat = self.update_fn(agg_flat, old_flat)  # [B*T*N, D]
        updated = updated_flat.reshape(B, T, N, D)

        # Zero out invalid objects
        updated = updated * valid_mask.unsqueeze(1).unsqueeze(-1).float()

        return updated


class ForceAwareGNN(nn.Module):
    """Multi-layer force-aware GNN for object interaction modeling."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_attention: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.edge_network = ForceAwareEdgeNetwork(state_dim=16, edge_dim=edge_dim)
        self.layers = nn.ModuleList([
            GNNSingleLayer(node_dim, edge_dim, hidden_dim, dropout, use_attention)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_dim)
            for _ in range(num_layers)
        ])

    @staticmethod
    def _has_valid_neighbor(valid_mask: torch.Tensor) -> torch.Tensor:
        """Return [B, N] mask for nodes with at least one non-self valid neighbor."""
        B, N = valid_mask.shape
        pair_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
        eye = torch.eye(N, device=valid_mask.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        return pair_mask.any(dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,        # [B, T, N, D]
        dyn_state: torch.Tensor,      # [B, T, N, 16] - raw state for edge construction
        force_matrix: torch.Tensor,   # [B, T, N, N, 3]
        valid_mask: torch.Tensor,     # [B, N]
    ) -> torch.Tensor:
        """
        Returns:
            interacted_tokens: [B, T, N, D]
        """
        edge_feat = self.edge_network(dyn_state, force_matrix, valid_mask)

        h = tokens
        valid = valid_mask.unsqueeze(1).unsqueeze(-1).float()
        has_neighbor = self._has_valid_neighbor(valid_mask).unsqueeze(1).unsqueeze(-1)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_feat, valid_mask)
            if self.use_residual:
                candidate = norm(h + h_new) * valid
            else:
                candidate = h_new
            h = torch.where(has_neighbor, candidate, h * valid)

        return h
