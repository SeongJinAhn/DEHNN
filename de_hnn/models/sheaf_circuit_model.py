"""
Sheaf Hypergraph Neural Network for Circuit Performance Prediction.

Implements:
  1. SheafBuilder — learns restriction maps F_{v←e} for each (node, hyperedge)
  2. SheafHyperConv — one layer of sheaf Laplacian diffusion on hypergraphs
  3. SheafCircuitModel — full model with encoder, sheaf conv stack, readout, MLP head

The sheaf Laplacian is:  L = delta^T @ D_e^{-1} @ delta
where delta is the coboundary map built from the learned restriction maps.
Diffusion:  H' = H - sigma * L @ H   (explicit Euler step)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import scatter


# ── Sheaf Builder ────────────────────────────────────────────────────────────

class SheafBuilder(nn.Module):
    """
    Learns d×d restriction maps F_{v←e} for each node-hyperedge incidence.

    Given node features h_v and a hyperedge feature (mean of member nodes),
    produces a d×d matrix via an MLP.
    """

    def __init__(self, input_dim, stalk_dim, hidden_dim=64):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim * stalk_dim),
        )

    def forward(self, x, node_indices, hedge_indices, num_hedges):
        """
        Args:
            x: [N, input_dim] node features
            node_indices: [M] node ids for each incidence
            hedge_indices: [M] hyperedge ids for each incidence
            num_hedges: int

        Returns:
            maps: [M, d, d] restriction maps for each incidence
        """
        d = self.stalk_dim

        # Compute hyperedge features as mean of member node features
        hedge_feat = scatter(x[node_indices], hedge_indices, dim=0,
                             dim_size=num_hedges, reduce="mean")  # [E, input_dim]

        # For each incidence (v, e), concatenate node feat and hedge feat
        pair_feat = torch.cat([x[node_indices], hedge_feat[hedge_indices]], dim=-1)  # [M, 2*input_dim]

        maps = self.mlp(pair_feat).view(-1, d, d)  # [M, d, d]
        return maps


# ── Sheaf Hypergraph Convolution ─────────────────────────────────────────────

class SheafHyperConv(nn.Module):
    """
    One layer of sheaf hypergraph diffusion.

    Given stalk features  H ∈ R^{N×d}  and restriction maps  F_{v←e},
    computes:
        H' = H - sigma * L_sheaf @ H

    where L_sheaf = delta^T @ D_e^{-1} @ delta  is the sheaf Laplacian
    and delta is the coboundary operator.

    For efficiency we compute L @ H directly without materializing L:
        (L @ H)_v = sum_{e ∋ v} (1/|e|) * F_{v←e}^T @ sum_{u ∈ e} F_{u←e} @ H_u
                   aggregated appropriately.
    """

    def __init__(self, stalk_dim, sigma=1.0):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.sigma = nn.Parameter(torch.tensor(sigma))

    def forward(self, H, maps, node_indices, hedge_indices, num_nodes, num_hedges):
        """
        Args:
            H: [N, d] stalk features
            maps: [M, d, d] restriction maps
            node_indices: [M] node ids
            hedge_indices: [M] hyperedge ids
            num_nodes: int
            num_hedges: int

        Returns:
            H_new: [N, d] updated stalk features
        """
        d = self.stalk_dim

        # Step 1: Compute F_{v←e} @ H_v for each incidence → projected features
        # maps: [M, d, d], H[node_indices]: [M, d]
        Fh = torch.bmm(maps, H[node_indices].unsqueeze(-1)).squeeze(-1)  # [M, d]

        # Step 2: Sum projected features per hyperedge → aggregate per edge
        hedge_sum = scatter(Fh, hedge_indices, dim=0, dim_size=num_hedges,
                            reduce="sum")  # [E, d]

        # Step 3: Compute hyperedge degree (number of members) for normalization
        ones = torch.ones(node_indices.size(0), device=H.device)
        hedge_deg = scatter(ones, hedge_indices, dim=0, dim_size=num_hedges,
                            reduce="sum")  # [E]
        hedge_deg = hedge_deg.clamp(min=1.0)

        # Normalize: hedge_sum / |e|
        hedge_msg = hedge_sum / hedge_deg.unsqueeze(-1)  # [E, d]

        # Step 4: For each incidence, compute F_{v←e}^T @ hedge_msg_e
        # maps^T: [M, d, d], hedge_msg[hedge_indices]: [M, d]
        maps_t = maps.transpose(1, 2)  # [M, d, d]
        back_msg = torch.bmm(maps_t, hedge_msg[hedge_indices].unsqueeze(-1)).squeeze(-1)  # [M, d]

        # Step 5: Aggregate back to nodes
        node_agg = scatter(back_msg, node_indices, dim=0, dim_size=num_nodes,
                           reduce="sum")  # [N, d]

        # Step 6: Also compute the diagonal term: sum_{e ∋ v} (1/|e|) F^T F H_v
        FtF_Hv = torch.bmm(maps_t, Fh.unsqueeze(-1)).squeeze(-1)  # [M, d]
        diag_scale = (1.0 / hedge_deg[hedge_indices]).unsqueeze(-1)  # [M, 1]
        diag_term = scatter(FtF_Hv * diag_scale, node_indices, dim=0,
                            dim_size=num_nodes, reduce="sum")  # [N, d]

        # Sheaf Laplacian: L @ H = diag_term - node_agg
        # (This is: sum_{e∋v} 1/|e| [F^T F H_v - F^T (sum_u F H_u / |e|)])
        # Simplified: the diffusion update
        LH = diag_term - node_agg

        H_new = H - self.sigma * LH
        return H_new


# ── Full Model ───────────────────────────────────────────────────────────────

class SheafCircuitModel(nn.Module):
    """
    Sheaf Hypergraph Neural Network for graph-level circuit prediction.

    Architecture:
        Node encoder → [SheafBuilder + SheafHyperConv + Norm + ReLU] × L → Readout → MLP → targets
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        stalk_dim=8,
        num_layers=4,
        output_dim=4,      # [gain, bw, pm, fom]
        dropout=0.1,
        sigma=1.0,
        readout="mean",
    ):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout

        # Node encoder: input_dim → hidden_dim → stalk_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim),
        )

        # Sheaf convolution layers
        self.sheaf_builders = nn.ModuleList()
        self.sheaf_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lins = nn.ModuleList()  # post-diffusion linear

        for _ in range(num_layers):
            self.sheaf_builders.append(SheafBuilder(stalk_dim, stalk_dim, hidden_dim))
            self.sheaf_convs.append(SheafHyperConv(stalk_dim, sigma))
            self.norms.append(nn.LayerNorm(stalk_dim))
            self.lins.append(nn.Linear(stalk_dim, stalk_dim))

        # MLP head for graph-level prediction
        self.head = nn.Sequential(
            nn.Linear(stalk_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, data, device=None):
        """
        Args:
            data: PyG Batch with x, node_indices, hedge_indices, num_hedges, batch
        Returns:
            pred: [B, output_dim] predictions
        """
        if device is None:
            device = data.x.device

        x = data.x.to(device)
        node_indices = data.node_indices.to(device)
        hedge_indices = data.hedge_indices.to(device)
        batch = data.batch.to(device)
        num_nodes = x.size(0)

        # Compute num_hedges for the full batch
        # In batched mode, hedge_indices are already offset by PyG
        num_hedges = int(hedge_indices.max().item()) + 1 if hedge_indices.numel() > 0 else 0

        # Encode
        H = self.encoder(x)  # [N, stalk_dim]

        # Sheaf diffusion layers
        for i in range(self.num_layers):
            maps = self.sheaf_builders[i](H, node_indices, hedge_indices, num_hedges)
            H_diff = self.sheaf_convs[i](H, maps, node_indices, hedge_indices,
                                          num_nodes, num_hedges)
            H_diff = self.lins[i](H_diff)
            H_diff = self.norms[i](H_diff)
            H = H + F.relu(H_diff)  # residual
            H = F.dropout(H, p=self.dropout, training=self.training)

        # Graph-level readout
        if self.readout_type == "mean":
            graph_emb = global_mean_pool(H, batch)
        elif self.readout_type == "add":
            graph_emb = global_add_pool(H, batch)
        else:
            graph_emb = global_mean_pool(H, batch)

        # Predict
        pred = self.head(graph_emb)
        return pred


# ── Simple GNN Baseline ──────────────────────────────────────────────────────

class GNNBaseline(nn.Module):
    """
    Simple GCN-style baseline for comparison.
    Uses the original DAG edge_index (not hypergraph).
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=4,
                 output_dim=4, dropout=0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = dropout

    def forward(self, data, device=None):
        if device is None:
            device = data.x.device

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = data.batch.to(device)

        h = self.encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, edge_index)))
            h = F.dropout(h, p=self.dropout, training=self.training)

        graph_emb = global_mean_pool(h, batch)
        return self.head(graph_emb)
