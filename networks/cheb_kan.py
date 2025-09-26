import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    

class ChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, generator=None):
        super(ChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.base_activation = torch.nn.SiLU()
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, 
                        mean=0.0, 
                        std=1 / (input_dim * (degree + 1)), 
                        generator=generator)
        
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.base_weight, 
                        mean=0.0, std=1 / input_dim, 
                        generator=generator)
        self.register_buffer("arange", torch.arange(0, degree + 1, 1)[None, None, :])

    def forward(self, x):
        y_base = F.linear(self.base_activation(x), self.base_weight, bias=None)
        x = torch.tanh(x)
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)

        x = x.acos()
        x = x[..., None] * self.arange
        x = x.cos()

        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)

        return y.view(-1, self.outdim) + y_base


class ChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree,
        gen=None,
        mult=0.2
    ):
        super(ChebyKAN, self).__init__()
        self.mult = mult
        self.layers = torch.nn.ModuleList()
        i = 0
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                ChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    generator=gen
                )
            )
            if i == 0:
                self.layers[-1].base_activation = nn.Identity()
            i += 1
        
        
    def forward(self, x: torch.Tensor, t=None):
        for layer in self.layers:
            x = layer(x)
        return x * self.mult
    

class AChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_cfg, generator=None):
        super(AChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.base_degrees = init_cfg[0]
        self.topk = init_cfg[1]
        self.base_activation = torch.nn.SiLU()
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, 
                        mean=0.0, 
                        std=1 / (input_dim * (degree + 1)), 
                        generator=generator)
        
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.base_weight, 
                        mean=0.0, 
                        std=1 / input_dim, 
                        generator=generator)
        self.register_buffer("arange", torch.arange(0, self.base_degrees + 1, 1))
        self.register_buffer("gating_weights", torch.ones(degree + 1))

        self.n_experts = degree - self.base_degrees
        self.logits = None

    def forward(self, x, t, logits):
        # Apply base weight
        y_base = F.linear(self.base_activation(x), self.base_weight, bias=None)
        # y_base = 0
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        if abs(t) > 1e-8:
            noise = torch.randn(self.n_experts, device=x.device) * t * 0.3
        else:
            noise = 0
        
        self.logits = logits
        _, topk_indices = torch.topk(self.logits + noise, k=self.topk)
        topk_values = F.sigmoid(self.logits[topk_indices])
        topk_indices_added = topk_indices + self.base_degrees + 1
        curr_basis_choice = torch.cat((self.arange, topk_indices_added), dim=0)
        curr_gating_weights = torch.scatter(self.gating_weights, 
                                            0, topk_indices_added, topk_values)
        x = torch.tanh(x)
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)

        x = x.acos()
        x = x[..., None] * curr_basis_choice[None, None, :]
        x = x.cos()
        gated_cheby_coeffs = self.cheby_coeffs * curr_gating_weights
        curr_cheby_coeffs = gated_cheby_coeffs[:, :, curr_basis_choice]
        y = torch.einsum(
            "bid,iod->bo", x, curr_cheby_coeffs
        )  # shape = (batch_size, outdim)
        return y.view(-1, self.outdim) + y_base

  
class AChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree,
        init_cfg,
        gen=None,
        mult=0.2
    ):
        super(AChebyKAN, self).__init__()

        self.mult = mult
        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                AChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,
                    init_cfg=init_cfg,
                    generator=gen
                )
            )
        
        d = 128
        buffer = torch.randn(d)
        self.register_buffer("z", buffer)
        self.n_experts = self.layers[0].n_experts
        self.deep_prior = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, (len(layers_hidden) - 1) * self.n_experts)
        )

    def forward(self, x: torch.Tensor, t):
        logits = self.deep_prior(self.z)
        for i, layer in enumerate(self.layers):
            curr_logits = logits[i * self.n_experts:(i + 1) * self.n_experts]
            x = layer(x, t, curr_logits)
        return x * self.mult
    
class RandChebyKANLinear(nn.Module):
    def __init__(self, input_dim, output_dim, degree, init_cfg, generator=None):
        super(RandChebyKANLinear, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.base_degrees = init_cfg[0]
        self.topk = init_cfg[1]
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, 
                                                     output_dim, 
                                                     self.base_degrees + self.topk + 1))
        nn.init.normal_(self.cheby_coeffs, 
                        mean=0.0, 
                        std=1 / (input_dim * (self.base_degrees + self.topk + 1) * 5), 
                        generator=generator)
        self.cheby_coeffs.data[:, :, 0] = 0

        
        self.base_activation = torch.nn.SiLU()
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.base_weight, 
                        mean=0.0, 
                        std=1 / input_dim, 
                        generator=generator)

        rand_degrees = np.random.choice(np.arange(self.base_degrees + 1, degree + 1), 
                                        size=self.topk, 
                                        replace=False)
        rand_degrees.sort()
        arange = torch.cat(
            [torch.arange(0, self.base_degrees + 1, 1), torch.tensor(rand_degrees)], dim=0)
        self.register_buffer("arange", arange[None, None, :])

    def forward(self, x, t=None):
        y_base = F.linear(self.base_activation(x), self.base_weight, bias=None)

        x = torch.tanh(x)
        x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        x = x.acos()
        x = x[..., None] * self.arange
        x = x.cos()
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        return y.view(-1, self.outdim) + y_base


class RandChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree,
        init_cfg,
        gen=None,
        mult=0.2
    ):
        super(RandChebyKAN, self).__init__()

        self.mult = mult
        self.layers = torch.nn.ModuleList()
        i = 0
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                RandChebyKANLinear(
                    in_features,
                    out_features,
                    degree=degree,\
                    init_cfg=init_cfg,
                    generator=gen
                )
            )

            if i == 0:
                self.layers[-1].base_activation = nn.Identity()
                self.layers[-1].base_weight.data.uniform_(-1/3, 1/3)
                # pass
            i += 1

    def forward(self, x: torch.Tensor, t=None):
        for layer in self.layers:
            x = layer(x)
        return x * self.mult