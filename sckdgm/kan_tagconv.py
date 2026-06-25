import math

import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor


class KANLinear(nn.Module):
    """Fourier KAN linear map used inside K-hop TAG message passing."""

    def __init__(self, input_dim, output_dim, grid_size=4, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        scale = math.sqrt(input_dim * grid_size)
        self.fourier_coeffs = nn.Parameter(torch.randn(2, output_dim, input_dim, grid_size) / scale)
        self.bias = nn.Parameter(torch.zeros(1, output_dim)) if bias else None

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_dim * self.grid_size)
        nn.init.normal_(self.fourier_coeffs, std=std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        batch_size = x.shape[0]
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
        x = x.view(batch_size, 1, self.input_dim, 1) / 10.0
        cos_kx = (torch.cos(k * x) - 1.0).view(1, batch_size, self.input_dim, self.grid_size)
        sin_kx = torch.sin(k * x).view(1, batch_size, self.input_dim, self.grid_size)
        out = torch.einsum("dbik,djik->bj", torch.cat([cos_kx, sin_kx], dim=0), self.fourier_coeffs)
        if self.bias is not None:
            out = out + self.bias
        return out


class KANTAGConv(MessagePassing):
    """TAGConv with a KAN transform for each hop."""

    def __init__(self, in_channels, out_channels, K=3, grid_size=4, normalize=True):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize
        self.kan_layers = nn.ModuleList(
            [KANLinear(in_channels, out_channels, grid_size=grid_size, bias=False) for _ in range(K + 1)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(in_channels) if i > 0 else nn.Identity() for i in range(K + 1)])
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.kan_layers:
            layer.reset_parameters()
        for norm in self.layer_norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                x.size(0),
                improved=False,
                add_self_loops=False,
                dtype=x.dtype,
            )

        out = self.kan_layers[0](x)
        x_cur = x
        for hop in range(1, self.K + 1):
            x_cur = self.layer_norms[hop](x_cur)
            x_cur = self.propagate(edge_index, x=x_cur, edge_weight=edge_weight)
            out = out + self.kan_layers[hop](x_cur)
        return out + self.bias

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
