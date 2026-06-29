"""
AnalogGenie → Terminal-Hypergraph data pipeline.

Converts SPICE netlists from the AnalogGenie dataset into PyG Data objects
with hypergraph incidence structure and terminal role annotations.

Input:  AnalogGenie Dataset/ directory with {id}/{id}.cir + Port{id}.txt
Output: List of PyG Data objects with:
    - x:             [N, feat_dim] node features (one-hot role + device type)
    - node_indices:  [M] node IDs for each incidence pair
    - hedge_indices: [M] hyperedge IDs for each incidence pair
    - num_hedges:    int, number of hyperedges (nets)
    - tau:           [N] terminal role index per node
    - y:             int, circuit type label
    - edge_index:    [2, E] original pin-level graph edges (for GNN baseline)
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data
from collections import defaultdict


# Terminal role vocabulary
ROLE_TO_IDX = {
    'gate': 0,       # MOSFET gate (high-impedance input)
    'drain': 1,      # MOSFET drain
    'source': 2,     # MOSFET source
    'bulk': 3,       # MOSFET bulk/body
    'collector': 4,  # BJT collector
    'base': 5,       # BJT base
    'emitter': 6,    # BJT emitter
    'pos': 7,        # passive positive terminal
    'neg': 8,        # passive negative terminal
    'port': 9,       # circuit I/O port
    'device': 10,    # device center node (NM1, PM1, etc.)
    'digital_io': 11,  # digital gate I/O
}
NUM_ROLES = len(ROLE_TO_IDX)

# Device type vocabulary
DEVICE_TO_IDX = {
    'nmos4': 0, 'pmos4': 1,
    'npn': 2, 'pnp': 3,
    'resistor': 4, 'capacitor': 5, 'inductor': 6, 'diode': 7,
    'XOR': 8, 'PFD': 9, 'INVERTER': 10, 'TRANSMISSION_GATE': 11,
    'port': 12,
}
NUM_DEVICE_TYPES = len(DEVICE_TO_IDX)

# Circuit type labels based on AnalogGenie data_categorization.md (ID ranges)
CIRCUIT_LABELS = {
    'amplifier': 0,
    'current_mirror': 1,
    'opamp': 2,
    'oscillator_vco': 3,
    'pll': 4,
    'lna': 5,
    'mixer': 6,
    'power_amplifier': 7,
    'comparator': 8,
    'power_converter': 9,
    'bandgap': 10,
    'ldo': 11,
    'sc_sampler': 12,
    'filter': 13,
    'other': 14,
}

# Terminal suffixes → role mapping
SUFFIX_TO_ROLE = {
    '_G': 'gate', '_D': 'drain', '_S': 'source', '_B': 'bulk',
    '_C': 'collector', '_E': 'emitter',
    '_P': 'pos', '_N': 'neg',
    '_A': 'digital_io', '_Y': 'digital_io', '_Q': 'digital_io',
    '_QA': 'digital_io', '_QB': 'digital_io',
    '_VDD': 'port', '_VSS': 'port',
}

# Device terminal definitions (matching SPICE2GRAPH_full.py)
DEVICE_TERMINALS = {
    'nmos4': ['_D', '_G', '_S', '_B'],
    'pmos4': ['_D', '_G', '_S', '_B'],
    'npn': ['_C', '_B', '_E'],
    'pnp': ['_C', '_B', '_E'],
    'resistor': ['_P', '_N'],
    'capacitor': ['_P', '_N'],
    'inductor': ['_P', '_N'],
    'diode': ['_P', '_N'],
    'XOR': ['_A', '_B', '_VDD', '_VSS', '_Y'],
    'PFD': ['_A', '_B', '_QA', '_QB', '_VDD', '_VSS'],
    'INVERTER': ['_A', '_Q', '_VDD', '_VSS'],
    'TRANSMISSION_GATE': ['_A', '_B', '_C', '_VDD', '_VSS'],
}

DEVICE_PREFIXES = {
    'nmos4': 'NM', 'pmos4': 'PM',
    'npn': 'NPN', 'pnp': 'PNP',
    'resistor': 'R', 'capacitor': 'C', 'inductor': 'L', 'diode': 'DIO',
    'XOR': 'XOR', 'PFD': 'PFD', 'INVERTER': 'INVERTER',
    'TRANSMISSION_GATE': 'TRANSMISSION_GATE',
}


def read_netlist(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    netlist = []
    for line in lines:
        parts = line.strip().replace('(', '').replace(')', '').split()
        if parts:
            netlist.append(parts)
    return netlist


def read_ports(filename):
    with open(filename, 'r') as f:
        ports = f.readline().strip().split()
    return ports


def get_circuit_label(circuit_id):
    """Assign circuit type label based on ID range."""
    cid = int(circuit_id)
    if 1045 <= cid <= 1080:
        return CIRCUIT_LABELS['comparator']
    elif 1081 <= cid <= 1090:
        return CIRCUIT_LABELS['lna']
    elif 1091 <= cid <= 1099:
        return CIRCUIT_LABELS['mixer']
    elif 1100 <= cid <= 1108:
        return CIRCUIT_LABELS['power_amplifier']
    elif 1109 <= cid <= 1190:
        return CIRCUIT_LABELS['oscillator_vco']
    elif 1191 <= cid <= 1460:
        return CIRCUIT_LABELS['power_converter']
    elif 1461 <= cid <= 1780:
        return CIRCUIT_LABELS['bandgap']
    elif 1781 <= cid <= 2180:
        return CIRCUIT_LABELS['opamp']
    elif 2181 <= cid <= 2630:
        return CIRCUIT_LABELS['ldo']
    elif 2631 <= cid <= 3502:
        return CIRCUIT_LABELS['sc_sampler']
    else:
        return CIRCUIT_LABELS['other']


def get_role(node_name, is_port=False):
    """Determine terminal role from node name."""
    if is_port:
        return 'port'
    for suffix, role in SUFFIX_TO_ROLE.items():
        if node_name.endswith(suffix):
            return role
    # BJT _B suffix conflicts with MOSFET _B; check context
    if node_name.endswith('_B'):
        for prefix in ['NPN', 'PNP']:
            if node_name.startswith(prefix):
                return 'base'
        return 'bulk'
    return 'device'


def spice_to_terminal_hypergraph(netlist, ports):
    """
    Convert SPICE netlist to terminal-hypergraph.

    Returns:
        nodes: list of node names
        node_features: dict {node_name: (role_idx, device_type_idx)}
        hyperedges: list of sets, each set contains node indices in that net
        edge_pairs: list of (src, dst) for original pin-level edges
    """
    nodes = []
    node_to_idx = {}
    node_meta = {}  # node_name -> (role, device_type)

    # Add port nodes
    for p in ports:
        idx = len(nodes)
        nodes.append(p)
        node_to_idx[p] = idx
        node_meta[p] = ('port', 'port')

    # Counters for device naming (matching SPICE2GRAPH_full.py)
    counters = defaultdict(lambda: 1)

    # Parse netlist: build device nodes and track net connections
    net_dict = defaultdict(set)  # net_name -> set of node indices
    device_nodes = []  # (device_center_name, device_type, terminal_names, net_names)

    for component in netlist:
        comp_type = component[-1]
        if comp_type not in DEVICE_TERMINALS:
            continue

        idx_counter = counters[comp_type]
        counters[comp_type] += 1
        prefix = DEVICE_PREFIXES[comp_type]
        terminals = DEVICE_TERMINALS[comp_type]

        # Device center node
        center_name = f'{prefix}{idx_counter}'
        center_idx = len(nodes)
        nodes.append(center_name)
        node_to_idx[center_name] = center_idx
        node_meta[center_name] = ('device', comp_type)

        # Terminal nodes
        terminal_node_names = []
        for suffix in terminals:
            term_name = f'{center_name}{suffix}'
            term_idx = len(nodes)
            nodes.append(term_name)
            node_to_idx[term_name] = term_idx
            role = get_role(term_name)
            node_meta[term_name] = (role, comp_type)
            terminal_node_names.append(term_name)

        # Net connections from netlist: component format is [inst_name, net1, net2, ..., type]
        net_names = component[1:-1]
        for term_name, net_name in zip(terminal_node_names, net_names):
            term_idx = node_to_idx[term_name]
            if net_name in node_to_idx:
                # Direct connection to a port
                net_dict[net_name].add(term_idx)
                net_dict[net_name].add(node_to_idx[net_name])
            else:
                # Internal net
                net_dict[net_name].add(term_idx)

    # Build hyperedges: each net is a hyperedge
    hyperedges = []
    for net_name, member_indices in net_dict.items():
        if len(member_indices) >= 2:
            hyperedges.append(member_indices)

    # Build original edge pairs (device center ↔ its terminals)
    edge_pairs = []
    counters2 = defaultdict(lambda: 1)
    for component in netlist:
        comp_type = component[-1]
        if comp_type not in DEVICE_TERMINALS:
            continue
        idx_counter = counters2[comp_type]
        counters2[comp_type] += 1
        prefix = DEVICE_PREFIXES[comp_type]
        center_name = f'{prefix}{idx_counter}'
        center_idx = node_to_idx[center_name]
        for suffix in DEVICE_TERMINALS[comp_type]:
            term_name = f'{center_name}{suffix}'
            term_idx = node_to_idx[term_name]
            edge_pairs.append((center_idx, term_idx))
            edge_pairs.append((term_idx, center_idx))

    # Also add clique edges within each net (for GNN baseline)
    for members in hyperedges:
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                edge_pairs.append((members_list[i], members_list[j]))
                edge_pairs.append((members_list[j], members_list[i]))

    return nodes, node_meta, hyperedges, edge_pairs


def build_pyg_data(nodes, node_meta, hyperedges, edge_pairs, circuit_id):
    """Convert parsed circuit to PyG Data object."""
    num_nodes = len(nodes)

    # Node features: one-hot role + one-hot device type
    tau = torch.zeros(num_nodes, dtype=torch.long)
    device_type = torch.zeros(num_nodes, dtype=torch.long)

    for i, name in enumerate(nodes):
        role_str, dev_str = node_meta[name]
        tau[i] = ROLE_TO_IDX.get(role_str, ROLE_TO_IDX['device'])
        device_type[i] = DEVICE_TO_IDX.get(dev_str, DEVICE_TO_IDX['port'])

    # One-hot features
    role_onehot = torch.zeros(num_nodes, NUM_ROLES)
    role_onehot.scatter_(1, tau.unsqueeze(1), 1.0)
    dev_onehot = torch.zeros(num_nodes, NUM_DEVICE_TYPES)
    dev_onehot.scatter_(1, device_type.unsqueeze(1), 1.0)
    x = torch.cat([role_onehot, dev_onehot], dim=1)  # [N, NUM_ROLES + NUM_DEVICE_TYPES]

    # Hypergraph incidence: (node_indices, hedge_indices) pairs
    node_indices_list = []
    hedge_indices_list = []
    for eid, members in enumerate(hyperedges):
        for nid in members:
            node_indices_list.append(nid)
            hedge_indices_list.append(eid)

    node_indices = torch.tensor(node_indices_list, dtype=torch.long)
    hedge_indices = torch.tensor(hedge_indices_list, dtype=torch.long)
    num_hedges = len(hyperedges)

    # Original graph edges (for GNN baseline)
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        # Remove duplicates
        edge_index = torch.unique(edge_index, dim=1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Label
    y = torch.tensor([get_circuit_label(circuit_id)], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        node_indices=node_indices,
        hedge_indices=hedge_indices,
        num_hedges=num_hedges,
        tau=tau,
        device_type=device_type,
        y=y,
        circuit_id=circuit_id,
        num_nodes=num_nodes,
    )
    return data


def load_analoggenie_dataset(data_dir, id_range=None, verbose=True):
    """
    Load AnalogGenie circuits and convert to terminal-hypergraph PyG data.

    Args:
        data_dir: path to AnalogGenie Dataset/ directory
        id_range: tuple (start, end) of circuit IDs to load, or None for all
        verbose: print progress

    Returns:
        dataset: list of PyG Data objects
    """
    if id_range is None:
        id_range = (1, 3502)

    dataset = []
    skipped = 0

    for cid in range(id_range[0], id_range[1] + 1):
        cid_str = str(cid)
        netlist_file = os.path.join(data_dir, cid_str, f'{cid_str}.cir')
        port_file = os.path.join(data_dir, cid_str, f'Port{cid_str}.txt')

        if not os.path.isfile(netlist_file) or not os.path.isfile(port_file):
            skipped += 1
            continue

        try:
            netlist = read_netlist(netlist_file)
            ports = read_ports(port_file)
            nodes, node_meta, hyperedges, edge_pairs = spice_to_terminal_hypergraph(netlist, ports)

            if len(nodes) < 3 or len(hyperedges) < 1:
                skipped += 1
                continue

            data = build_pyg_data(nodes, node_meta, hyperedges, edge_pairs, cid_str)
            dataset.append(data)
        except Exception as e:
            if verbose:
                print(f"  Skipping circuit {cid}: {e}")
            skipped += 1
            continue

    if verbose:
        print(f"Loaded {len(dataset)} circuits, skipped {skipped}")
        if dataset:
            d = dataset[0]
            print(f"  Sample: nodes={d.num_nodes}, hedges={d.num_hedges}, "
                  f"incidences={d.node_indices.shape[0]}, edges={d.edge_index.shape[1]}")

    return dataset


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "Dataset"
    dataset = load_analoggenie_dataset(data_dir)

    # Print stats
    labels = [d.y.item() for d in dataset]
    from collections import Counter
    label_counts = Counter(labels)
    inv_labels = {v: k for k, v in CIRCUIT_LABELS.items()}
    print("\nLabel distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"  {inv_labels.get(label_id, '?'):20s}: {count}")

    # Print role distribution
    all_tau = torch.cat([d.tau for d in dataset])
    inv_roles = {v: k for k, v in ROLE_TO_IDX.items()}
    print("\nRole distribution:")
    for role_id in range(NUM_ROLES):
        count = (all_tau == role_id).sum().item()
        if count > 0:
            print(f"  {inv_roles[role_id]:15s}: {count}")
