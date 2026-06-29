"""
Sheaf Hypergraph Neural Network for Circuit Performance Prediction.

Implements:
  1. SheafBuilder — learns restriction maps F_{v←e} for each (node, hyperedge)  [B3: generic]
  2. TypedRestriction — role-typed restriction maps (B1: typed, B2: physics-hard, B4: physics-soft)
  3. SheafHyperConv — one layer of sheaf Laplacian diffusion on hypergraphs
  4. SheafCircuitModel — full model with encoder, sheaf conv stack, readout, MLP head

Ablation modes for restriction maps:
  - "generic"  (B3): MLP(h_v, h_e) → d×d per incidence (Duta'23 style)
  - "typed"    (B1): R[tau(v)] → one d×d matrix per role type
  - "physics"  (B2): hard constraints — gate=low-rank proj, drain-source involution tie
  - "physics_soft" (B4): init with physics structure + regularizer, but learnable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import scatter


# ── Sheaf Builder (B3: Generic Learned) ─────────────────────────────────────

class SheafBuilder(nn.Module):
    """
    Learns d×d restriction maps F_{v←e} for each node-hyperedge incidence.
    This is the generic baseline (B3) — most expressive, no structural prior.
    """

    def __init__(self, input_dim, stalk_dim, hidden_dim=64):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim * stalk_dim),
        )

    def forward(self, x, node_indices, hedge_indices, num_hedges, tau=None):
        d = self.stalk_dim
        hedge_feat = scatter(x[node_indices], hedge_indices, dim=0,
                             dim_size=num_hedges, reduce="mean")
        pair_feat = torch.cat([x[node_indices], hedge_feat[hedge_indices]], dim=-1)
        maps = self.mlp(pair_feat).view(-1, d, d)
        return maps, torch.tensor(0.0, device=x.device)


# ── Typed Restriction (B1/B2/B4: Role-Structured) ──────────────────────────

class TypedRestriction(nn.Module):
    """
    Restriction maps structured by terminal role.

    Modes:
      "typed"        (B1): one free d×d matrix per role — no physics constraint
      "physics"      (B2): hard constraints:
                            - gate (role 0) = low-rank projection
                            - source (role 2) = involution-conjugate of drain (role 1)
                            - bulk (role 3) = small-norm
      "physics_soft" (B4): free matrices initialized with physics structure,
                            regularized toward physics during training
    """

    def __init__(self, d, num_roles, mode="typed", gate_rank=None):
        super().__init__()
        self.d = d
        self.mode = mode
        self.num_roles = num_roles
        self.gate_rank = gate_rank or max(1, d // 2)

        self.R = nn.Parameter(torch.stack([torch.eye(d) for _ in range(num_roles)]))

        if mode in ("physics", "physics_soft"):
            self.Pg = nn.Parameter(torch.randn(d, self.gate_rank) * 0.1)
            sigma = torch.zeros(d, d)
            for i in range(d):
                sigma[i, d - 1 - i] = 1.0
            self.register_buffer('sigma', sigma)

    def _physics_matrices(self):
        R = self.R.clone()
        # Gate (role 0): low-rank projection (high-impedance input)
        Pg = self.Pg
        PgTPg = Pg.T @ Pg
        PgTPg_inv = torch.linalg.pinv(PgTPg)
        proj_g = Pg @ PgTPg_inv @ Pg.T
        R[0] = proj_g

        # Source (role 2) = sigma @ R_drain @ sigma^T (conduction symmetry)
        if self.num_roles > 2:
            R[2] = self.sigma @ R[1] @ self.sigma.T

        # Bulk (role 3): weak coupling
        if self.num_roles > 3:
            R[3] = 0.1 * R[3]

        return R

    def _compute_reg(self):
        """Physics regularizer for soft mode."""
        Pg = self.Pg
        PgTPg_inv = torch.linalg.pinv(Pg.T @ Pg)
        proj_g = Pg @ PgTPg_inv @ Pg.T

        reg = (self.R[0] - proj_g.detach()).pow(2).sum()
        if self.num_roles > 2:
            target_s = self.sigma @ self.R[1].detach() @ self.sigma.T
            reg = reg + (self.R[2] - target_s).pow(2).sum()
        if self.num_roles > 3:
            reg = reg + (self.R[3].norm() - 0.1).pow(2)
        return reg

    def forward(self, tau, node_indices):
        """
        Args:
            tau: [N] role index for each node
            node_indices: [M] node IDs for each incidence

        Returns:
            maps: [M, d, d] per-incidence restriction maps
            reg: scalar regularization loss (0 for typed/physics modes)
        """
        if self.mode == "typed":
            R = self.R
            reg = torch.tensor(0.0, device=R.device)
        elif self.mode == "physics":
            R = self._physics_matrices()
            reg = torch.tensor(0.0, device=R.device)
        elif self.mode == "physics_soft":
            R = self.R
            reg = self._compute_reg()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        per_node_R = R[tau]                 # [N, d, d]
        maps = per_node_R[node_indices]     # [M, d, d]
        return maps, reg


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
        Node encoder → [SheafBuilder/TypedRestriction + SheafHyperConv + Norm + ReLU] × L
        → Readout → MLP → targets

    sheaf_mode controls the restriction map:
        "generic"      (B3): MLP-learned per incidence
        "typed"        (B1): one matrix per terminal role
        "physics"      (B2): hard physics constraints
        "physics_soft" (B4): soft physics constraints + regularizer
        "identity"     (B0): identity maps (no sheaf)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        stalk_dim=8,
        num_layers=4,
        output_dim=4,
        dropout=0.1,
        sigma=1.0,
        readout="mean",
        sheaf_mode="generic",
        num_roles=12,
        gate_rank=None,
    ):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.readout_type = readout
        self.sheaf_mode = sheaf_mode

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stalk_dim),
        )

        self.sheaf_builders = nn.ModuleList()
        self.sheaf_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lins = nn.ModuleList()

        for _ in range(num_layers):
            if sheaf_mode == "generic":
                self.sheaf_builders.append(SheafBuilder(stalk_dim, stalk_dim, hidden_dim))
            elif sheaf_mode in ("typed", "physics", "physics_soft"):
                self.sheaf_builders.append(TypedRestriction(stalk_dim, num_roles,
                                                            mode=sheaf_mode,
                                                            gate_rank=gate_rank))
            elif sheaf_mode == "identity":
                self.sheaf_builders.append(None)
            else:
                raise ValueError(f"Unknown sheaf_mode: {sheaf_mode}")

            self.sheaf_convs.append(SheafHyperConv(stalk_dim, sigma))
            self.norms.append(nn.LayerNorm(stalk_dim))
            self.lins.append(nn.Linear(stalk_dim, stalk_dim))

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
        Returns:
            pred: [B, output_dim] predictions
            reg_loss: scalar physics regularization (0 for non-physics modes)
        """
        if device is None:
            device = data.x.device

        x = data.x.to(device)
        node_indices = data.node_indices.to(device)
        hedge_indices = data.hedge_indices.to(device)
        batch = data.batch.to(device)
        tau = data.tau.to(device) if hasattr(data, 'tau') and data.tau is not None else None
        num_nodes = x.size(0)
        num_hedges = int(hedge_indices.max().item()) + 1 if hedge_indices.numel() > 0 else 0

        H = self.encoder(x)
        total_reg = torch.tensor(0.0, device=device)

        for i in range(self.num_layers):
            if self.sheaf_mode == "identity":
                # B0: identity restriction maps
                num_incidences = node_indices.size(0)
                maps = torch.eye(self.stalk_dim, device=device).unsqueeze(0).expand(num_incidences, -1, -1)
                reg = torch.tensor(0.0, device=device)
            elif self.sheaf_mode == "generic":
                maps, reg = self.sheaf_builders[i](H, node_indices, hedge_indices, num_hedges, tau)
            else:
                maps, reg = self.sheaf_builders[i](tau, node_indices)

            total_reg = total_reg + reg

            H_diff = self.sheaf_convs[i](H, maps, node_indices, hedge_indices,
                                          num_nodes, num_hedges)
            H_diff = self.lins[i](H_diff)
            H_diff = self.norms[i](H_diff)
            H = H + F.relu(H_diff)
            H = F.dropout(H, p=self.dropout, training=self.training)

        if self.readout_type == "mean":
            graph_emb = global_mean_pool(H, batch)
        elif self.readout_type == "add":
            graph_emb = global_add_pool(H, batch)
        else:
            graph_emb = global_mean_pool(H, batch)

        pred = self.head(graph_emb)
        return pred, total_reg


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
