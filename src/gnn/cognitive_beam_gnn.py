"""
Graph Attention Network for cognitive, multi-satellite beamforming.

A heterogeneous graph represents:
    - Satellite nodes – orbital state + current beam parameters.
    - Ground-station nodes – location + channel quality metrics.
    - Edges – active or candidate links between a satellite and a ground
      station (directed: sat → gs).

The GNN propagates information across the graph so that each satellite can
make coordinated beamforming decisions that account for neighbouring
interference and uneven traffic loads.

Reference:
    Velickovic et al. (2018) – Graph Attention Networks.
    Kipf & Welling (2017) – Semi-Supervised Classification with GCNs.

Note:
    This module requires the ``torch_geometric`` package.  If it is not
    installed, importing from ``gnn`` will raise an ``ImportError`` with a
    helpful message.
"""

try:
    import torch_geometric  # noqa: F401
    _TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    _TORCH_GEOMETRIC_AVAILABLE = False

if not _TORCH_GEOMETRIC_AVAILABLE:
    raise ImportError(
        "torch_geometric is required for the GNN module. "
        "Install it with: pip install torch-geometric"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class CognitiveBeamGNN(nn.Module):
    """
    Heterogeneous Graph Attention Network for multi-satellite beamforming.

    Nodes are split into two types – satellites and ground stations.
    Separate linear encoders project each node type into a shared latent
    space.  Two GAT convolution layers then refine the representations
    using attention over neighbouring nodes.  A final decoder maps each
    satellite's latent vector to a 4-dimensional beamforming action:
    (delta_phase, delta_power, mcs_index, rb_allocation).

    Args:
        node_features:  Number of input features per node (same for both
                        satellite and ground-station nodes).
        hidden:         Hidden/latent dimension after encoding.
        gat_heads:      Number of attention heads in each GAT layer.
        dropout:        Dropout probability applied after each GAT layer.
    """

    def __init__(
        self,
        node_features: int,
        hidden: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        # Type-specific input encoders
        self.sat_encoder = nn.Linear(node_features, hidden)
        self.gs_encoder = nn.Linear(node_features, hidden)

        # GAT convolution layers (bipartite: satellites attend to ground stations
        # and vice versa via the full node feature matrix)
        self.conv1 = GATConv(hidden, hidden, heads=gat_heads, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden, hidden, heads=gat_heads, concat=False, dropout=dropout)

        # Output head: beamforming action per satellite node
        self.beam_decoder = nn.Linear(hidden, 4)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.sat_encoder, self.gs_encoder, self.beam_decoder]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass over a heterogeneous graph.

        Args:
            data: A ``torch_geometric.data.HeteroData`` object with:
                  - ``data['sat'].x``               – (N_sat, node_features)
                  - ``data['ground_station'].x``     – (N_gs, node_features)
                  - ``data['sat', 'to', 'ground_station'].edge_index``
                    – (2, E) edge index (satellite → ground station)
                  - ``data['sat'].num_nodes``        – integer

        Returns:
            Beam action tensor of shape (N_sat, 4):
            [delta_phase, delta_power, mcs_index, rb_allocation].
        """
        x_sat = F.relu(self.sat_encoder(data["sat"].x))
        x_gs = F.relu(self.gs_encoder(data["ground_station"].x))

        # Concatenate into a single node matrix for homogeneous GAT convolution
        x = torch.cat([x_sat, x_gs], dim=0)

        edge_index = data["sat", "to", "ground_station"].edge_index

        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training)

        # Extract satellite node representations
        n_sat = data["sat"].num_nodes
        x_sat_out = x[:n_sat]

        beam_actions = self.beam_decoder(x_sat_out)
        return beam_actions
