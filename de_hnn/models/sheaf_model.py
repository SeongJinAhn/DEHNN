"""
Sheaf Hypergraph GNN for VLSI netlist node/net property prediction.

Drop-in replacement for model_att.py's GNN_node, using SheafBipartiteConv
instead of HyperConvLayer. Same interface: takes HeteroData, outputs
(node_pred, net_pred).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils.dropout import dropout_edge

import sys
sys.path.append("./layers/")
from sheaf_conv_layer import SheafBipartiteConv


class SheafGNN_node(nn.Module):
    """
    Sheaf Hypergraph GNN for node-level and net-level prediction.

    Same interface as GNN_node from model_att.py:
        forward(data, device) -> (h_inst, h_net)

    Args:
        num_layer: Number of sheaf conv layers (>= 2)
        emb_dim: Embedding dimension
        out_node_dim: Output dimension for node predictions
        out_net_dim: Output dimension for net predictions
        stalk_dim: Sheaf stalk dimension
        node_dim: Input node feature dimension
        net_dim: Input net feature dimension
        vn: Use virtual node
        trans: Use transformer for virtual node
        aggr: Aggregation type
        device: Device string
    """

    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim,
                 stalk_dim=4,
                 JK="concat", residual=True,
                 norm_type="layer", aggr="add",
                 node_dim=None, net_dim=None,
                 num_nodes=None, vn=False, trans=False,
                 device='cuda'):
        super().__init__()
        self.device = device
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim
        self.stalk_dim = stalk_dim
        self.vn = vn
        self.trans = trans

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.net_encoder = nn.Sequential(
            nn.Linear(net_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Virtual node (same as original)
        if self.vn:
            self.virtualnode_encoder = nn.Sequential(
                nn.Linear(node_dim * 2, emb_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(emb_dim * 2, emb_dim)
            )
            self.mlp_virtualnode_list = nn.ModuleList()
            self.back_virtualnode_list = nn.ModuleList()

            if self.trans:
                self.transformer_virtualnode_list = nn.ModuleList()

            for layer in range(num_layer):
                self.mlp_virtualnode_list.append(nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.LeakyReLU(),
                    nn.Linear(emb_dim, emb_dim)
                ))
                self.back_virtualnode_list.append(nn.Sequential(
                    nn.Linear(emb_dim * 2, emb_dim),
                    nn.LeakyReLU(),
                    nn.Linear(emb_dim, emb_dim)
                ))
                if self.trans:
                    self.transformer_virtualnode_list.append(
                        nn.TransformerEncoderLayer(
                            d_model=emb_dim * 2, nhead=8,
                            dim_feedforward=512
                        )
                    )

        # Sheaf conv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(
                SheafBipartiteConv(
                    emb_dim, emb_dim,
                    stalk_dim=stalk_dim, aggr=aggr
                )
            )
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(emb_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(emb_dim))

        # Output heads
        self.fc1_node = nn.Linear(emb_dim, 256)
        self.fc2_node = nn.Linear(256, out_node_dim)

        self.fc1_net = nn.Linear(emb_dim, 256)
        self.fc2_net = nn.Linear(256, out_net_dim)

    def forward(self, data, device):
        """
        Same interface as GNN_node.forward.

        Args:
            data: HeteroData with node/net features and edge indices
            device: torch device

        Returns:
            h_inst: [N, out_node_dim] node predictions
            h_net: [M, out_net_dim] net predictions
        """
        h_inst = data['node'].x.to(device)
        h_net = data['net'].x.to(device)

        edge_index_node_to_net = data['node', 'to', 'net'].edge_index
        edge_weight_node_to_net = data['node', 'to', 'net'].edge_weight
        edge_index_net_to_node = data['net', 'to', 'node'].edge_index
        edge_weight_net_to_node = data['net', 'to', 'node'].edge_weight

        # Edge dropout (same as original)
        edge_index_node_to_net, edge_mask = dropout_edge(
            edge_index_node_to_net, p=0.2
        )
        edge_index_net_to_node = edge_index_net_to_node[:, edge_mask]
        edge_weight_node_to_net = edge_weight_node_to_net[edge_mask]
        edge_weight_net_to_node = edge_weight_net_to_node[edge_mask]
        edge_type_node_to_net = data['node', 'to', 'net'].edge_type[edge_mask]

        # Encode
        h_inst = self.node_encoder(h_inst)
        h_net = self.net_encoder(h_net)

        if self.vn:
            batch = data.batch.to(device)
            virtualnode_embedding = self.virtualnode_encoder(data.vn.to(device))

        # Message passing layers
        for layer in range(self.num_layer):
            if self.vn:
                h_inst = self.back_virtualnode_list[layer](
                    torch.concat([h_inst, virtualnode_embedding[batch]], dim=1)
                ) + h_inst

            h_inst, h_net = self.convs[layer](
                h_inst, h_net,
                edge_index_node_to_net, edge_weight_node_to_net,
                edge_type_node_to_net,
                edge_index_net_to_node, edge_weight_net_to_node,
                device
            )
            h_inst = self.norms[layer](h_inst)
            h_net = self.norms[layer](h_net)
            h_inst = nn.functional.leaky_relu(h_inst)
            h_net = nn.functional.leaky_relu(h_net)

            if (layer < self.num_layer - 1) and self.vn:
                vn_temp = torch.concat([
                    global_mean_pool(h_inst, batch),
                    global_max_pool(h_inst, batch)
                ], dim=1)
                if self.trans:
                    vn_temp = self.transformer_virtualnode_list[layer](vn_temp)
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](vn_temp) + virtualnode_embedding
                else:
                    virtualnode_embedding = self.mlp_virtualnode_list[layer](vn_temp) + virtualnode_embedding

        # Output
        h_inst = self.fc2_node(nn.functional.leaky_relu(self.fc1_node(h_inst)))
        h_net = self.fc2_net(nn.functional.leaky_relu(self.fc1_net(h_net)))
        return h_inst, h_net
