"""Quick smoke test for new model architectures (Sheaf ODE, Cell Complex)."""

import torch
import sys
sys.path.append("models/layers/")
sys.path.append("data/")

from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from models.model_att import GNN_node
from data.cell_complex_utils import build_cell_complex

device = 'cpu'

def make_synthetic_data(num_nodes=100, num_nets=50, avg_degree=4, with_cell_complex=False):
    """Create synthetic circuit netlist data for testing."""
    # Random bipartite edges
    num_edges = num_nodes * avg_degree
    node_idx = torch.randint(0, num_nodes, (num_edges,))
    net_idx = torch.randint(0, num_nets, (num_edges,))

    # ~20% source, 80% sink (typical circuit pattern)
    edge_type = torch.zeros(num_edges, dtype=torch.bool)
    edge_type[:num_edges // 5] = True

    # Separate source and sink edges
    source_edge = torch.stack([node_idx[edge_type], net_idx[edge_type]])
    sink_edge = torch.stack([node_idx[~edge_type], net_idx[~edge_type]])

    edge_index = torch.cat([sink_edge, source_edge], dim=1)
    edge_type_all = torch.cat([
        torch.zeros(sink_edge.shape[1], dtype=torch.bool),
        torch.ones(source_edge.shape[1], dtype=torch.bool)
    ])

    h_data = HeteroData()
    h_data['node'].x = torch.randn(num_nodes, 7)  # node features
    h_data['net'].x = torch.randn(num_nets, 3)     # net features

    h_data['node', 'to', 'net'].edge_index, h_data['node', 'to', 'net'].edge_weight = gcn_norm(edge_index, add_self_loops=False)
    h_data['node', 'to', 'net'].edge_type = edge_type_all
    h_data['net', 'to', 'node'].edge_index, h_data['net', 'to', 'node'].edge_weight = gcn_norm(edge_index.flip(0), add_self_loops=False)
    h_data.num_instances = num_nodes

    # Virtual node data
    h_data.batch = torch.zeros(num_nodes, dtype=torch.long)
    h_data.num_vn = 1
    h_data.vn = torch.randn(1, 14)  # 7*2

    if with_cell_complex:
        cc_data = build_cell_complex(
            h_data['node', 'to', 'net'].edge_index,
            h_data['node', 'to', 'net'].edge_type,
            num_nodes, num_nets
        )
        h_data['cell_complex'] = cc_data
        print(f"  Cell complex: {cc_data['num_1cells']} 1-cells, "
              f"B1: {cc_data['B1_index'].shape}, B2: {cc_data['B2_index'].shape}")

    return h_data


def test_model(gnn_type, **kwargs):
    """Test a single model type."""
    print(f"\n{'='*60}")
    print(f"Testing gnn_type='{gnn_type}'")
    print(f"{'='*60}")

    with_cc = gnn_type in ['cell_complex', 'cell_complex_att']
    data = make_synthetic_data(with_cell_complex=with_cc)

    model = GNN_node(
        num_layer=3, emb_dim=16, out_node_dim=1, out_net_dim=1,
        node_dim=7, net_dim=3, gnn_type=gnn_type,
        vn=False, trans=False, aggr='add', JK='Normal',
        device=device, **kwargs
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Forward pass
    node_out, net_out = model(data, device)
    print(f"  Node output: {node_out.shape}")
    print(f"  Net output: {net_out.shape}")

    # Backward pass
    loss = node_out.sum() + net_out.sum()
    loss.backward()
    print(f"  Backward pass: OK (loss={loss.item():.4f})")

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_count = sum(1 for _ in model.parameters())
    print(f"  Gradients: {grad_count}/{total_count} parameters have grads")

    print(f"  PASSED!")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Smoke test for DE-HNN new architectures")
    print("="*60)

    results = {}

    # Test existing models first (regression check)
    for gnn_type in ['dehnn', 'dehnn_att', 'digcn', 'digat']:
        try:
            results[gnn_type] = test_model(gnn_type)
        except Exception as e:
            print(f"  FAILED: {e}")
            results[gnn_type] = False

    # Test Sheaf ODE
    try:
        results['sheaf_ode'] = test_model('sheaf_ode', ode_T=1.0, ode_tol=1e-2, ode_method='euler')
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        results['sheaf_ode'] = False

    # Test Cell Complex
    for gnn_type in ['cell_complex', 'cell_complex_att']:
        try:
            results[gnn_type] = test_model(gnn_type)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[gnn_type] = False

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} [{status}]")

    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
