"""Shared-trace neural displacement fields minimizing bulk and spring energy.

The geometry, kinematics and nondimensionalization are those of the mixed PINN.
Only displacement is learned; stress is C:epsilon(u). Essential conditions are
hard constraints. Free tractions, equilibrium and spring laws are natural weak
conditions. Fixed hidden features give a linear least-squares energy minimizer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from partitioned_mixed_pinn import (Case, PatchNet, ROOT, evaluate, gauss,
                                    quadrature, sample_one, traction)
from junction_enrichment import JunctionModes
from local_neural_enrichment import LocalNeuralModes, adapt_local_features
from local_virtual_work import displacement_residual

import numpy as np
from scipy import linalg
import torch
from torch import nn

class EnergyNet(PatchNet):
    def __init__(self, case, width=24, seed=7, displacement_scaling="compliance",
                 corner_enrichment=False, elastic_corners=False, junction_radius=None,
                 local_neural=None, additional_local=None):
        super().__init__(case, width, seed, "shared_trace", displacement_scaling,
                         corner_enrichment, elastic_corners)
        self.full_domain = False
        self.weights[-1] = nn.Parameter(self.weights[-1][..., :2].detach().clone())
        self.biases[-1] = nn.Parameter(self.biases[-1][..., :2].detach().clone())
        self.junction = (JunctionModes(case, self.cells, self.vein_width/2, junction_radius)
                         if junction_radius is not None else None)
        self.local = (LocalNeuralModes(case, self.cells, self.vein_width/2, **local_neural)
                      if local_neural is not None else None)
        self.additional_local = (LocalNeuralModes(case, self.cells, self.vein_width/2, **additional_local)
                                 if additional_local is not None else None)

    def displacement(self, xy, homogeneous=False, include_enrichment=True):
        uv = super().forward(xy, include_corners=include_enrichment,
                             homogeneous=homogeneous, displacement_only=True)
        if self.junction is not None and include_enrichment:
            uv = uv+self.junction(xy)
        if self.local is not None and include_enrichment:
            uv = uv+self.local(xy)
        if self.additional_local is not None and include_enrichment:
            uv = uv+self.additional_local(xy)
        return uv

    def displacement_strain(self, xy, homogeneous=False, include_enrichment=True):
        uv = self.displacement(xy, homogeneous, include_enrichment)
        du, dv = [torch.autograd.grad(uv[..., j].sum(), xy, create_graph=True)[0]
                  for j in range(2)]
        eps = torch.stack([du[..., 0], dv[..., 1], du[..., 1]+dv[..., 0]], -1)
        return uv, eps

    def stress(self, eps):
        nu = self.case.poisson
        lam = self.E[..., 0]*nu/((1+nu)*(1-2*nu))
        mu = self.E[..., 0]/(2*(1+nu))
        dilation = eps[..., :2].sum(-1)
        return torch.stack([lam*dilation+2*mu*eps[..., 0],
                            lam*dilation+2*mu*eps[..., 1], mu*eps[..., 2]], -1)

    def forward(self, xy):
        # Existing field exporters use no_grad for sampling; constitutive
        # stresses still require coordinate derivatives inside that context.
        track = torch.is_grad_enabled()
        with torch.enable_grad():
            if not xy.requires_grad:
                xy = xy.detach().requires_grad_(True)
            uv, eps = self.displacement_strain(xy)
            result = torch.cat([uv, self.stress(eps)], -1)
        return result if track else result.detach()

    def fields(self, xy):
        uv, eps = self.displacement_strain(xy)
        stress = self.stress(eps)
        ds = [torch.autograd.grad(stress[..., j].sum(), xy, create_graph=True)[0]
              for j in range(3)]
        div = torch.stack([ds[0][..., 0]+ds[2][..., 1],
                           ds[2][..., 0]+ds[1][..., 1]], -1)
        return torch.cat([uv, stress], -1), eps, eps, div

    def readout_parameters(self):
        linear = [self.weights[-1], self.biases[-1], self.trace.node_values,
                  self.trace.w2, self.trace.b2]
        return linear+self.enrichment_parameters()

    def enrichment_parameters(self):
        linear = []
        if self.elastic_corners:
            linear.append(self.elastic_corner_coeff)
        elif self.corner_enrichment:
            linear.append(self.displacement_corner_coeff)
        if self.junction is not None:
            linear.append(self.junction.coefficients)
        if self.local is not None:
            linear.append(self.local.coefficients)
        if self.additional_local is not None:
            linear.append(self.additional_local.coefficients)
        return linear

    def readout_displacement(self, parameter):
        """Select the displacement component while all other readouts are zero."""
        for component in (self.junction, self.local, self.additional_local):
            if component is not None and parameter is component.coefficients:
                return component
        include_enrichment = any(parameter is value for value in self.enrichment_parameters())
        return lambda xy: self.displacement(xy, True, include_enrichment)


def load_model(path):
    saved = torch.load(path, weights_only=False)
    if saved["kinematics"] != "shared_trace":
        raise ValueError("Energy minimization requires the shared displacement traces.")
    model = EnergyNet(Case(**saved["case"]), saved["width"], saved["seed"],
                      saved["displacement_scaling"], saved["corner_enrichment"],
                      saved["elastic_corners"], saved.get("junction_radius"),
                      saved.get("local_neural"), saved.get("additional_local"))
    state = saved["state_dict"].copy()
    # Mixed checkpoints share every displacement coefficient and hidden feature.
    for key in ("weights.2", "biases.2"):
        state[key] = state[key][..., :2].clone()
    model.load_state_dict(state)
    return model, saved["seed"]


def energy_residual(model, quad, homogeneous=False, include_enrichment=True):
    """r(u) with total potential Pi(u) = ||r(u)||^2 / 2, no body/traction load."""
    displacement = lambda xy: model.displacement(xy, homogeneous, include_enrichment)
    return displacement_residual(model, displacement, quad)


def fit_energy_readout(model, quad, rcond=None, extra_residual=None, equation_constraints=None,
                       interior_elimination=None):
    """Fit energy with optional residuals, equalities and local interior elimination."""
    def residual_of(displacement):
        energy = displacement_residual(model, displacement, quad)
        if extra_residual is None:
            return energy
        return torch.cat([energy, extra_residual(displacement)])

    interior_field = None

    def complete_displacement(xy):
        value = model.displacement(xy)
        return value if interior_field is None else value+interior_field(xy)

    def complete_residual():
        return residual_of(complete_displacement)

    linear = model.readout_parameters()
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    original = [p.detach().clone() for p in linear]
    initial_energy = energy_residual(model, quad).detach().numpy()
    before = float(.5*initial_energy@initial_energy)
    objective_before = before if extra_residual is None else before+float(
        .5*extra_residual(model.displacement).square().sum().detach())
    start = time.time()
    for p in parameters:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            for p in linear:
                p.zero_()
        base = residual_of(lambda xy: model.displacement(xy, False, False)).detach().numpy()
        constraint_base = (None if equation_constraints is None else
                           equation_constraints(lambda xy: model.displacement(xy, False, False)).numpy())
        # Support partitions pad shorter cells with zero integration weights.
        # Those rows are identically zero for every displacement; omit them
        # from the dense solve while retaining the batched field evaluation.
        rows = np.ones(len(base), dtype=bool)
        rows[:3*quad[1].numel()] = quad[1].detach().numpy().reshape(-1).repeat(3) > 0.
        energy_rows = int(rows[:len(initial_energy)].sum())
        base = base[rows]
        matrix = np.empty((len(base), sum(p.numel() for p in linear)), order="F")
        constraints = (None if constraint_base is None else
                       np.empty((len(constraint_base), matrix.shape[1]), order="F"))
        column = 0
        for p in linear:
            flat = p.reshape(-1)
            displacement = model.readout_displacement(p)
            for j in range(p.numel()):
                flat[j] = 1.
                matrix[:, column] = residual_of(displacement).detach().numpy()[rows]
                if constraints is not None:
                    constraints[:, column] = equation_constraints(displacement).numpy()
                flat[j] = 0.
                column += 1
                if column % 128 == 0 and j+1 < p.numel():
                    print(json.dumps({"energy_columns": column, "total_columns": matrix.shape[1],
                                      "seconds": round(time.time()-start, 1)}), flush=True)
            print(json.dumps({"energy_columns": column, "total_columns": matrix.shape[1],
                              "seconds": round(time.time()-start, 1)}), flush=True)
        if interior_elimination is not None:
            interior_elimination.condense(matrix, base, rows)
        scales = np.linalg.norm(matrix, axis=0)
        active = scales > 0.
        # Compact and normalize in the existing Fortran buffer. Boolean
        # indexing followed by division allocated two extra dense matrices.
        for target, source in enumerate(np.flatnonzero(active)):
            np.divide(matrix[:, source], scales[source], out=matrix[:, target])
        if constraints is not None:
            constraints = constraints[:, active]/scales[active]
        scaled = matrix[:, :int(active.sum())]
        cutoff = rcond if rcond is not None else np.finfo(matrix.dtype).eps*max(scaled.shape)
        if constraints is None:
            solution, _, rank, singular = linalg.lstsq(scaled, -base, cond=cutoff, lapack_driver="gelsd")
            predicted = base+scaled@solution
            solve_details = {"rank": int(rank), "retained_condition": float(singular[0]/singular[rank-1])}
        else:
            from equation_moments import solve_constrained
            solution, predicted, solve_details = solve_constrained(
                scaled, base, constraints, constraint_base, cutoff)
        coefficients = np.zeros(matrix.shape[1])
        coefficients[active] = solution/scales[active]
        if interior_elimination is not None:
            interior_field = interior_elimination.recover(coefficients)
        offset = 0
        with torch.no_grad():
            for p in linear:
                p.copy_(torch.from_numpy(coefficients[offset:offset+p.numel()]).reshape_as(p))
                offset += p.numel()
        residual = complete_residual().detach().numpy()[rows]
        result = {"solver": "svd" if constraints is None else "svd_nullspace", "energy_before": before,
                "energy_after": float(.5*residual[:energy_rows]@residual[:energy_rows]),
                "rows": matrix.shape[0], "columns": matrix.shape[1],
                "active_columns": int(active.sum()), "rcond": float(cutoff), **solve_details,
                "maximum_coefficient": float(np.abs(coefficients).max()),
                "matrix_field_residual_difference": float(np.linalg.norm(residual-predicted)),
                "elapsed_s": time.time()-start}
        if constraints is None:
            result["normalized_column_stationarity_max"] = float(np.abs(scaled.T@residual).max())
        else:
            actual_constraints = equation_constraints(complete_displacement).numpy()
            result["constraint_field_residual_max"] = float(np.abs(actual_constraints).max())
            result["matrix_field_constraint_difference"] = float(np.linalg.norm(
                actual_constraints-(constraint_base+constraints@solution)))
        if extra_residual is None and equation_constraints is None:
            result["load_derivative_reaction_full" if model.full_domain
                   else "load_derivative_reaction_quarter"] = float(
                       (1 if model.full_domain else 2)*base@residual)
        elif extra_residual is not None:
            result.update({"objective_before": objective_before,
                           "objective_after": float(.5*residual@residual),
                           "equation_residual_penalty_after": float(
                               .5*residual[energy_rows:]@residual[energy_rows:])})
        if interior_elimination is not None:
            result["interior_elimination"] = interior_elimination.summary()
        return result
    except BaseException:
        with torch.no_grad():
            for p, value in zip(linear, original):
                p.copy_(value)
        raise
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def energy_response(model, nodes, edge_nodes, region_nodes):
    response, fields, edges, sections = evaluate(model, nodes, edge_nodes, region_nodes)
    twice = response["bulk_twice_energy"]+response["interface_twice_energy"]
    # For shortening alpha/2 at the top, Pi(alpha)=alpha^2 Pi(1) gives R=4Pi.
    # This discrete energy derivative is reported beside the boundary stress
    # integral, not used as an independent verification of the same energy.
    response["stored_energy"] = .5*twice
    response["energy_reaction_quarter"] = 2*twice
    response["energy_axial_modulus_MPa"] = 4*model.case.soil_E_MPa*twice
    return response, fields, edges, sections


def junction_resultants(model, n=64):
    """One-sided transfer over the junction 5 mm, middle and tip 5 mm."""
    if model.case.geometry != "cross":
        return []
    z, w = gauss(n)
    halfw = model.vein_width/2
    segment_length = 5/model.case.specimen_size_mm
    rows = []
    for ie, edge in enumerate(model.edges):
        a, b, ax, fa = [edge[k] for k in ("a", "b", "axis", "fa")]
        if not edge["interface"] or model.cells[a]["material"] != "grout":
            continue
        bounds = model.cells[a]["bounds"]
        lo, hi = (bounds[2], bounds[3]) if ax == 0 else (bounds[0], bounds[1])
        fixed = bounds[fa]
        if abs(fixed-halfw) > 1e-12 or abs(lo-halfw) > 1e-12:
            continue
        for label, start, end in [("junction_5mm", lo, lo+segment_length),
                                  ("middle", lo+segment_length, hi-segment_length),
                                  ("tip_5mm", hi-segment_length, hi)]:
            s = start+(end-start)*z
            points = np.column_stack([np.full(n, fixed), s]) if ax == 0 else np.column_stack([s, np.full(n, fixed)])
            pa, pb = sample_one(model, a, points), sample_one(model, b, points)
            k = model.k.numpy() if ax == 0 else model.k.numpy()[::-1]
            integrate = lambda v: ((end-start)*w[:, None]*v).sum(0).tolist()
            rows.append({"edge": ie, "segment": label, "coordinate_bounds": [start, end],
                         "grout_resultant": integrate(traction(pa, ax)),
                         "soil_resultant": integrate(traction(pb, ax)),
                         "spring_resultant": integrate(k*(pb[:, :2]-pa[:, :2]))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--edge-nodes", type=int, default=48)
    parser.add_argument("--rcond", type=float)
    parser.add_argument("--check-nodes", type=int, default=60)
    parser.add_argument("--check-edge-nodes", type=int, default=192)
    parser.add_argument("--name", required=True)
    parser.add_argument("--add-elastic-corners", action="store_true")
    parser.add_argument("--junction-radius", type=float)
    parser.add_argument("--local-radius", type=float)
    parser.add_argument("--local-width", type=int, default=12)
    parser.add_argument("--local-steps", type=int, default=0)
    args = parser.parse_args()
    model, seed = load_model(args.resume)
    if args.add_elastic_corners and not model.elastic_corners:
        if model.corner_enrichment:
            raise ValueError("Start logarithmic elastic modes from a checkpoint without corner enrichment.")
        enriched = EnergyNet(model.case, model.width, seed, model.displacement_scaling,
                             elastic_corners=True,
                             junction_radius=model.junction.radius if model.junction else None,
                             local_neural=model.local.metadata() if model.local else None,
                             additional_local=model.additional_local.metadata() if model.additional_local else None)
        state = enriched.state_dict()
        state.update(model.state_dict())
        enriched.load_state_dict(state)
        model = enriched
    if args.junction_radius is not None:
        if model.junction is not None:
            raise ValueError("Resume the unenriched parent when changing junction support.")
        model.junction = JunctionModes(model.case, model.cells, model.vein_width/2,
                                       args.junction_radius)
    if args.local_radius is not None:
        if model.local is not None:
            raise ValueError("Resume the parent without local neural modes when changing support.")
        model.local = LocalNeuralModes(model.case, model.cells, model.vein_width/2,
                                       args.local_radius, args.local_width)
    adaptation = (adapt_local_features(model, args.nodes, args.edge_nodes, args.local_steps)
                  if args.local_steps else None)
    quad = list(quadrature(model.cells, args.nodes, args.edge_nodes))
    quad[0].requires_grad_(True)
    info = fit_energy_readout(model, quad, args.rcond)
    info.update({"nodes_per_axis": args.nodes, "edge_nodes": args.edge_nodes,
                 "hidden_features": "local_adaptation" if adaptation else "fixed_from_parent",
                 "local_adaptation": adaptation})
    response, fields, edges, sections = energy_response(model, args.nodes, args.edge_nodes, 24)
    independent = energy_response(model, args.check_nodes, args.check_edge_nodes, 48)[0]
    metadata = {"case": asdict(model.case), "width": model.width, "seed": seed,
                "kinematics": model.kinematics,
                "displacement_scaling": model.displacement_scaling,
                "corner_enrichment": model.corner_enrichment, "elastic_corners": model.elastic_corners,
                "junction_radius": model.junction.radius if model.junction else None,
                "local_neural": model.local.metadata() if model.local else None,
                "additional_local": model.additional_local.metadata() if model.additional_local else None,
                "formulation": "displacement_energy"}
    stem = ROOT/"results/feasibility/partitioned_2d"/args.name
    fields.to_csv(stem.with_suffix(".csv"), index=False)
    edges.to_csv(stem.with_name(stem.name+"_interfaces.csv"), index=False)
    sections.to_csv(stem.with_name(stem.name+"_sections.csv"), index=False)
    payload = {**metadata, "parent_checkpoint": str(args.resume), "optimization": info,
               "plane_condition": "plane_strain", "cells": model.cells, "edges": model.edges,
               "results": response,
               "junction_transfer": {"nodes_per_segment": 64, "segments": junction_resultants(model)},
               "independent_integration": {"nodes": args.check_nodes,
                   "edge_nodes": args.check_edge_nodes, "region_nodes": 48, "results": independent}}
    if model.local is not None:
        from local_virtual_work import probe_work
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        payload["local_virtual_work"] = probe_work(model)
        payload["junction_transfer"].update({"independent_nodes_per_segment": 128,
                                             "independent_segments": junction_resultants(model, 128)})
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({**metadata, "state_dict": model.state_dict()}, stem.with_suffix(".pt"))
    print(json.dumps({"name": args.name, "optimization": info,
                      "results": {k: independent[k] for k in (
                          "axial_modulus_MPa", "energy_axial_modulus_MPa", "regions",
                          "interface_resultants", "twice_stored_energy_over_boundary_work")}}, indent=2))


if __name__ == "__main__":
    main()
