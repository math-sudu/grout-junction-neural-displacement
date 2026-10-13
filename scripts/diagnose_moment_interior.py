"""Test admissible interior variations while retaining all equation moments.

The parent displacement is frozen. C1 rectangle bubbles vanish with their
gradient on every moment-rectangle boundary, so the full objective separates
into small independent interior fits. No observed forces enter the fit.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from partitioned_mixed_pinn import ROOT, gauss

import numpy as np
from scipy import linalg
import torch

from equation_moments import EquationMoments
from full_domain_pinn import evaluate_full, full_domain_quadrature, load_full_model, section_readouts
from local_virtual_work import strain_of, strain_residual
from partitioned_energy_pinn import energy_residual
from run_method_study import Tee, digest, save_json
from traction_control import EquilibriumControl, junction_scale_readout, stress_divergence


def bubble_basis(xy, bounds, degree):
    """Squared tensor bubble times total-degree Legendre polynomials."""
    x0, x1, y0, y1 = bounds
    a = ((xy[..., 0]-x0)/(x1-x0)).clamp(0, 1)
    b = ((xy[..., 1]-y0)/(y1-y0)).clamp(0, 1)
    axes = []
    for z in (2*a-1, 2*b-1):
        values = [torch.ones_like(z)]
        if degree:
            values.append(z)
        for n in range(1, degree):
            values.append(((2*n+1)*z*values[-1]-n*values[-2])/(n+1))
        axes.append(values)
    bubble = (16*a*(1-a)*b*(1-b)).square()
    return torch.stack([bubble*axes[0][i]*axes[1][j]
                        for i in range(degree+1) for j in range(degree+1-i)], -1)


class MomentInteriorField:
    def __init__(self, rectangles, degree, coefficients=None):
        self.rectangles, self.degree = rectangles, degree
        count = (degree+1)*(degree+2)//2
        self.coefficients = (torch.zeros(len(rectangles), count, 2) if coefficients is None
                             else torch.tensor(coefficients))

    def __call__(self, xy):
        value = 0*xy.square()
        for rectangle, coefficients in zip(self.rectangles, self.coefficients):
            cell = rectangle["cell"]
            local = bubble_basis(xy[cell], rectangle["bounds"], self.degree)@coefficients
            value = value.index_add(0, torch.tensor([cell]), local[None])
        return value

    def export(self):
        return {"rectangles": self.rectangles, "degree": self.degree,
                "coefficients": self.coefficients.tolist()}


def attach_interior(model, field):
    parent_displacement = model.displacement
    def displacement(xy, homogeneous=False, include_enrichment=True):
        value = parent_displacement(xy, homogeneous, include_enrichment)
        return value+field(xy) if include_enrichment else value
    model.displacement = displacement


def load_probe_field(path):
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    checkpoint = saved["execution"].get("neural_checkpoint_ref", saved["execution"]["parent_checkpoint_ref"])
    model = load_full_model(ROOT/checkpoint)
    attach_interior(model, MomentInteriorField(**saved["interior_field"]))
    return model


def local_operator(points, weights, bounds, degree, young, poisson, length=None):
    """Weighted strain-energy or divergence columns, ordered (feature, u/v)."""
    xy = torch.tensor(points, requires_grad=True)
    basis = bubble_basis(xy, bounds, degree)
    lam = young*poisson/((1+poisson)*(1-2*poisson))
    mu = young/(2*(1+poisson))
    columns = []
    for k in range(basis.shape[-1]):
        first = torch.autograd.grad(basis[:, k].sum(), xy, create_graph=True)[0]
        dx, dy = first[:, 0], first[:, 1]
        if length is None:
            u = torch.stack([(lam+mu)**.5*dx, mu**.5*dx, mu**.5*dy], -1)
            v = torch.stack([(lam+mu)**.5*dy, -mu**.5*dy, mu**.5*dx], -1)
        else:
            dxx, dxy = torch.autograd.grad(dx.sum(), xy, retain_graph=True)[0].unbind(-1)
            _, dyy = torch.autograd.grad(dy.sum(), xy, retain_graph=True)[0].unbind(-1)
            u = length*torch.stack([(lam+2*mu)*dxx+mu*dyy, (lam+mu)*dxy], -1)
            v = length*torch.stack([(lam+mu)*dxy, mu*dxx+(lam+2*mu)*dyy], -1)
        columns.extend([(weights[:, None]**.5*value.detach().numpy()).reshape(-1) for value in (u, v)])
    return np.column_stack(columns)


def rectangle_mask(points, weights, bounds):
    x0, x1, y0, y1 = bounds
    return ((points[:, 0] > x0) & (points[:, 0] < x1)
            & (points[:, 1] > y0) & (points[:, 1] < y1) & (weights > 0))


def fit_interior(model, energy_quad, control, moments, degree, rcond):
    if control.equilibrium_metric != "reference":
        raise ValueError("This diagnosis uses the parent's common-reference equilibrium metric.")
    rectangles = [{key: row[key] for key in ("cell", "material", "bounds")}
                  for row in moments.rows if row["kind"] == "bulk_balance" and row["component"] == 0]
    field = MomentInteriorField(rectangles, degree)
    xy, weights = energy_quad[:2]
    energy_base = strain_residual(model, strain_of(model.displacement(xy), xy), weights)
    energy_base = energy_base.detach().numpy().reshape(len(model.cells), -1, 3)
    xy, weights = control.quad[:2]
    stress = model.stress(strain_of(model.displacement(xy), xy))
    bulk_base = control.length*weights.sqrt()[..., None]*stress_divergence(stress, xy)
    bulk_base = bulk_base.detach().numpy()
    samples = [(q[0].detach().numpy(), q[1].numpy(), base)
               for q, base in ((energy_quad, energy_base), (control.quad, bulk_base))]
    rows = []
    for k, rectangle in enumerate(rectangles):
        cell, bounds = rectangle["cell"], rectangle["bounds"]
        matrices, bases, counts = [], [], []
        for (xy, weights, base), length in zip(samples, (None, control.length)):
            mask = rectangle_mask(xy[cell], weights[cell], bounds)
            counts.append(int(mask.sum()))
            if not mask.any():
                raise ValueError(f"No integration points in moment rectangle {rectangle}.")
            matrices.append(local_operator(xy[cell, mask], weights[cell, mask], bounds, degree,
                                           float(model.E[cell, 0, 0]), model.case.poisson, length))
            bases.append(base[cell, mask].reshape(-1))
        matrix, base = np.vstack(matrices), np.concatenate(bases)
        norms = np.linalg.norm(matrix, axis=0)
        scaled = matrix/norms
        coefficients, _, rank, singular = linalg.lstsq(scaled, -base, cond=rcond, lapack_driver="gelsd")
        coefficients /= norms
        field.coefficients[k] = torch.tensor(coefficients.reshape(-1, 2))
        gradient = matrix.T@base
        changes = [float(b@(a@coefficients)+.5*np.square(a@coefficients).sum())
                   for a, b in zip(matrices, bases)]
        rows.append({**rectangle, "energy_points": counts[0], "bulk_points": counts[1],
                     "rank": int(rank), "columns": len(coefficients),
                     "retained_condition": float(singular[0]/singular[rank-1]),
                     "maximum_normalized_objective_work": float(np.abs(gradient/norms).max()),
                     "best_single_direction_objective_reduction": float(.5*np.square(gradient/norms).max()),
                     "joint_objective_reduction": -sum(changes),
                     "energy_change": changes[0], "bulk_penalty_change": changes[1]})
    return field, rows


def objective_readout(model, quad, control):
    energy = float(.5*energy_residual(model, quad).detach().square().sum())
    terms = control.readout()
    return {"energy": energy, "control": terms, "objective": energy+terms["total_penalty"]}


def section_force_changes(model, field, positions):
    """Constitutive axial force of each correction, split at its own support."""
    z, w = gauss(field.degree+5)
    nu = model.case.poisson
    scale = model.case.soil_E_MPa*model.case.applied_shortening_mm
    rows = []
    for rectangle, coefficients in zip(field.rectangles, field.coefficients):
        x0, x1, y0, y1 = rectangle["bounds"]
        young = float(model.E[rectangle["cell"], 0, 0])
        lam = young*nu/((1+nu)*(1-2*nu))
        mu = young/(2*(1+nu))
        changes = []
        for y in positions:
            force = 0.
            if y0 < y < y1:
                xy = torch.tensor(np.column_stack([x0+(x1-x0)*z, np.full(len(z), y)]), requires_grad=True)
                value = bubble_basis(xy, rectangle["bounds"], field.degree)@coefficients
                strain = strain_of(value, xy)
                syy = lam*strain[:, 0]+(lam+2*mu)*strain[:, 1]
                force = -scale*(x1-x0)*float((syy.detach()*torch.tensor(w)).sum())
            changes.append(force)
        rows.append(changes)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/method_moments_full.json"))
    parser.add_argument("--degree", type=int, default=3)
    args = parser.parse_args()
    if args.degree < 0:
        parser.error("--degree must be nonnegative")
    checkpoint, output = ROOT/args.checkpoint, ROOT/args.output
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    parent = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    model = load_full_model(checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    settings, control_settings = config["full_domain"], config["equilibrium_control"]
    quad = list(full_domain_quadrature(model, settings["nodes"], settings["edge_nodes"], settings["support_nodes"]))
    quad[0].requires_grad_(True)
    def control(check=False):
        prefix = "check_" if check else ""
        return EquilibriumControl(model, control_settings[prefix+"nodes"],
                                  control_settings[prefix+"edge_nodes"], control_settings[prefix+"support_nodes"],
                                  model.vein_width*control_settings["length_over_arm_width"],
                                  control_settings["interface_metric"], control_settings["equilibrium_metric"])
    training, independent = control(), control(True)
    moment_options = {key: value for key, value in config["equation_moments"].items() if key != "check_edge_nodes"}
    moments = EquationMoments(model, **moment_options)
    trace = {"command": "python -X utf8 scripts/diagnose_moment_interior.py "+" ".join(sys.argv[1:]),
             "parent_checkpoint_ref": args.checkpoint.as_posix(), "parent_checkpoint_hash": digest(checkpoint),
             "config_ref": args.config.as_posix(), "started_at": datetime.now(timezone.utc).isoformat()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(json.dumps({"starting": output.name}), flush=True)
        before = objective_readout(model, quad, training)
        field, rows = fit_interior(model, quad, training, moments, args.degree, settings["rcond"])
        positions = parent["independent_readout"]["observation_supports"]["section_y"]
        for row, changes in zip(rows, section_force_changes(model, field, positions)):
            row["section_force_changes_N_per_mm"] = changes
        probe_moments = moments(field).numpy()
        attach_interior(model, field)
        print(json.dumps({"rectangles_fitted": len(rows),
                          "predicted_objective_reduction": sum(row["joint_objective_reduction"] for row in rows)}), flush=True)
        after = objective_readout(model, quad, training)
        checked = evaluate_full(model, settings["check_nodes"], settings["check_edge_nodes"],
                                settings["region_nodes"], settings["check_support_nodes"])
        finer_sections = section_readouts(model, checked["observation_supports"]["section_y"], 384)
        scale = checked["scales"]["line_resultant_N_per_mm_thickness"]
        force_changes = np.sum([row["section_force_changes_N_per_mm"] for row in rows], axis=0)
        parent_readout = parent["independent_readout"]
        neighboring = junction_scale_readout(model, [.375, .75], settings["check_edge_nodes"])
        verification = {
            "probe_moment_max_N_per_mm": float(np.abs(probe_moments).max())*scale,
            "objective_reduction_reconstruction_difference": abs(
                before["objective"]-after["objective"]-sum(row["joint_objective_reduction"] for row in rows)),
            "section_256_to_384_max_change_N_per_mm": max(
                abs(a["total_force"]-b["total_force"])*scale for a, b in zip(checked["sections"], finer_sections)),
            "section_attribution_max_difference_N_per_mm": max(
                abs((a["total_force"]-b["total_force"])*scale-change)
                for a, b, change in zip(checked["sections"], parent_readout["sections"], force_changes)),
            "original_interface_force_max_change_N_per_mm": max(
                abs(a-b)*scale for new, old in zip(checked["interface_segments"], parent_readout["interface_segments"])
                for side in ("a", "b", "spring")
                for a, b in zip(new["resultants_xy"][side], old["resultants_xy"][side])),
            "neighboring_interface_force_max_change_N_per_mm": max(
                abs(a-b)*scale for new_scale, old_scale in zip(neighboring, parent["neighboring_scale_readout"]["after"])
                for new, old in zip(new_scale["segments"], old_scale["segments"])
                for side in ("a", "b", "spring")
                for a, b in zip(new["resultants_xy"][side], old["resultants_xy"][side]))}
        payload = {"kind": "moment_preserving_interior_diagnosis", "execution": trace,
                   "question": "Can missing internal variations change the observed load path while retaining the existing moments and face fields?",
                   "interior_field": field.export(), "rectangle_fits": rows, "section_y_over_H": positions,
                   "training_objective": {"before": before, "after": after},
                   "independent_control": independent.readout(), "equation_moments": moments.readout(),
                   "parent_independent_readout": parent["independent_readout"], "independent_readout": checked,
                   "neighboring_scale_readout": neighboring,
                   "verification": verification}
        save_json(output.with_suffix(".json"), payload)
        replay = load_probe_field(output.with_suffix(".json"))
        replay_sections = section_readouts(replay, checked["observation_supports"]["section_y"], settings["check_edge_nodes"])
        verification["saved_field_section_replay_max_difference_N_per_mm"] = max(
            abs(a["total_force"]-b["total_force"])*scale for a, b in zip(checked["sections"], replay_sections))
        trace.update({"finished_at": datetime.now(timezone.utc).isoformat(), "exit_code": 0})
        save_json(output.with_suffix(".json"), payload)
        print(json.dumps({"completed": output.name, "verification": verification,
                          "section_forces_N_per_mm": [r["total_force"]*scale for r in checked["sections"]]}), flush=True)


if __name__ == "__main__":
    main()
