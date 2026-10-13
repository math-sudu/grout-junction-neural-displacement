"""Compact material-shared neural displacement correction at a cross junction."""
from __future__ import annotations

from partitioned_mixed_pinn import quadrature
from local_virtual_work import displacement_residual

import json
import time

import numpy as np
import torch
from torch import nn


class LocalNeuralModes(nn.Module):
    def __init__(self, case, cells, half_width, radius, width=12, seed=17):
        super().__init__()
        if case.geometry != "cross":
            raise ValueError("The local correction requires a cross junction.")
        if not 0. < radius < min(.5-half_width, case.vertical_span/2-half_width,
                                case.horizontal_span/2-half_width):
            raise ValueError("Local support must stay clear of the outer and tip boundaries.")
        self.radius, self.half_width, self.width, self.seed = radius, half_width, width, seed
        generator = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(1.5*torch.randn(2, 2, width, generator=generator))
        self.b1 = nn.Parameter(.5*torch.randn(2, 1, width, generator=generator))
        self.coefficients = nn.Parameter(torch.zeros(2, width+1, 2))
        self.register_buffer("material_ids", torch.tensor([int(c["material"] == "grout") for c in cells]))
        self.register_buffer("scale", torch.tensor([1., np.sqrt(case.soil_E_MPa/case.grout_E_MPa)]))

    def forward(self, xy):
        result = torch.zeros_like(xy)
        ids = self.material_ids
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            sign = torch.tensor([sx, sy])
            local = (xy*sign-self.half_width)/self.radius
            window = torch.relu(1-local.square().sum(-1, keepdim=True))**3
            hidden = torch.tanh(torch.bmm(local, self.w1[ids])+self.b1[ids])
            features = torch.cat([torch.ones_like(hidden[..., :1]), hidden], -1)
            result = result+window*torch.bmm(features, self.coefficients[ids])*sign
        return result*self.scale[ids, None, None]

    def metadata(self):
        return {"radius": self.radius, "width": self.width, "seed": self.seed}


def adapt_local_features(model, nodes, edge_nodes, steps):
    """Optimize only the local neural correction with the inherited field fixed."""
    from partitioned_energy_pinn import energy_residual
    if model.local is None:
        raise ValueError("Attach local neural modes before feature adaptation.")
    quad = list(quadrature(model.cells, nodes, edge_nodes))
    quad[0].requires_grad_(True)
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    local = model.local
    # Detach the inherited field once. Only the small added network is trained.
    local_initial = displacement_residual(model, local, quad).detach()
    base = (energy_residual(model, quad)-local_initial).detach()
    energy_scale = float(.5*local_initial.square().sum())
    if energy_scale == 0.:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        raise ValueError("Fit the local readout before adapting hidden features.")
    base_energy = float(.5*base.square().sum())
    for p in local.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.LBFGS(local.parameters(), lr=1., max_iter=steps,
                                  tolerance_grad=1e-10, tolerance_change=1e-13,
                                  history_size=50, line_search_fn="strong_wolfe")
    start, calls = time.time(), 0
    before = float(.5*(base+local_initial).square().sum())

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        residual = displacement_residual(model, local, quad)
        # Removing the fixed base energy and scaling by the correction's own
        # energy lets L-BFGS see this local problem despite a large background.
        increment = base@residual+.5*residual.square().sum()
        objective = increment/energy_scale
        objective.backward()
        calls += 1
        if calls % 50 == 0:
            print(json.dumps({"local_feature_calls": calls,
                              "energy": base_energy+float(increment.detach()),
                              "seconds": round(time.time()-start, 1)}), flush=True)
        return objective

    try:
        optimizer.step(closure)
        after = float(.5*(base+displacement_residual(model, local, quad)).square().sum().detach())
        return {"trainable": "local_neural_modes_only", "energy_before": before,
                "energy_after": after, "closure_calls": calls,
                "objective_energy_scale": energy_scale,
                "iterations": optimizer.state[next(iter(local.parameters()))]["n_iter"],
                "elapsed_s": time.time()-start}
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
