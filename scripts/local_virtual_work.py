"""Local admissible virtual displacements at the cross-junction.

The work is the first variation of the same bulk-plus-spring energy. A smooth
material-shared probe is split into a zero-trace patch bubble and its remainder
to distinguish missing interior variations from variations moving shared edges.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from partitioned_mixed_pinn import quadrature

import numpy as np
import torch


def strain_of(displacement, xy):
    du, dv = [torch.autograd.grad(displacement[..., j].sum(), xy,
                                 create_graph=True)[0] for j in range(2)]
    return torch.stack([du[..., 0], dv[..., 1], du[..., 1]+dv[..., 0]], -1)


def strain_residual(model, eps, weights):
    nu = model.case.poisson
    lam = model.E[..., 0]*nu/((1+nu)*(1-2*nu))
    mu = model.E[..., 0]/(2*(1+nu))
    root = torch.stack([(lam+mu).sqrt()*eps[..., :2].sum(-1),
                        mu.sqrt()*(eps[..., 0]-eps[..., 1]),
                        mu.sqrt()*eps[..., 2]], -1)
    return (weights.sqrt()[..., None]*root).flatten()


def displacement_residual(model, displacement, quad):
    xy, weights, faces, face_weights = quad
    bulk = strain_residual(model, strain_of(displacement(xy), xy), weights)
    uv = displacement(faces).reshape(len(model.cells), 4, -1, 2)
    residuals = [bulk]
    for edge in model.edges:
        if not edge["interface"]:
            continue
        a, b, axis, fa, fb = [edge[k] for k in ("a", "b", "axis", "fa", "fb")]
        k = model.k if axis == 0 else model.k.flip(0)
        residuals.append(((face_weights[a, fa, :, None]*k).sqrt()
                          *(uv[b, fb]-uv[a, fa])).flatten())
    return torch.cat(residuals)


def local_probe(model, xy, radius, material, component, mode, part):
    center = model.vein_width/2
    result = torch.zeros_like(xy[..., 0])
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        x = (sx*xy[..., 0]-center)/radius
        y = (sy*xy[..., 1]-center)/radius
        window = torch.relu(1-x*x-y*y)**3
        polynomial = {"constant": torch.ones_like(x), "x": x, "y": y, "xy": x*y}[mode]
        result = result+window*polynomial*(sx if component == 0 else sy)
    mask = torch.tensor([c["material"] == material for c in model.cells])[:, None]
    result = result*mask
    if part != "full":
        b = model.bounds
        x = (xy[..., 0]-b[..., 0])/(b[..., 1]-b[..., 0])
        y = (xy[..., 1]-b[..., 2])/(b[..., 3]-b[..., 2])
        bubble = 16*x*(1-x)*y*(1-y)
        result = result*(bubble if part == "bubble" else 1-bubble)
    unit = torch.tensor([1., 0.] if component == 0 else [0., 1.])
    return result[..., None]*unit


def probe_work(model, nodes=60, edge_nodes=192, radius=None):
    from partitioned_energy_pinn import energy_residual
    radius = model.vein_width if radius is None else radius
    quad = list(quadrature(model.cells, nodes, edge_nodes))
    quad[0].requires_grad_(True)
    field = energy_residual(model, quad).detach()
    bulk_size = quad[0].shape[0]*quad[0].shape[1]*3
    rows = []
    for material in ("grout", "soil"):
        for component in range(2):
            for mode in ("constant", "x", "y", "xy"):
                for part in ("full", "bubble", "trace"):
                    probe = lambda xy: local_probe(model, xy, radius, material, component, mode, part)
                    direction = displacement_residual(model, probe, quad).detach()
                    bulk = float(field[:bulk_size]@direction[:bulk_size])
                    spring = float(field[bulk_size:]@direction[bulk_size:])
                    stiffness = float(direction@direction)
                    rows.append({"material": material, "component": "xy"[component],
                                 "mode": mode, "part": part, "bulk_work": bulk,
                                 "spring_work": spring, "virtual_work": bulk+spring,
                                 "self_stiffness": stiffness,
                                 "work_over_energy_norm": (bulk+spring)/np.sqrt(stiffness)})
    return {"radius_over_size": radius, "nodes": nodes, "edge_nodes": edge_nodes,
            "probes": rows}


def main():
    from partitioned_energy_pinn import load_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, _ = load_model(args.checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = {"checkpoint": str(args.checkpoint), **probe_work(model)}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for row in sorted(result["probes"], key=lambda r: abs(r["work_over_energy_norm"]), reverse=True)[:12]:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
