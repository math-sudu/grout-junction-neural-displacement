"""Jointly fit neural readouts and C1 interiors under the retained moments.

Disjoint interior blocks are eliminated before the neural nullspace solve.
The result JSON and its neural checkpoint reconstruct the complete field with
diagnose_moment_interior.load_probe_field.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from partitioned_mixed_pinn import ROOT

import numpy as np
from scipy import linalg
import torch

from diagnose_moment_interior import (MomentInteriorField, attach_interior, load_probe_field,
                                      local_operator, objective_readout, rectangle_mask)
from equation_moments import EquationMoments
from full_domain_pinn import evaluate_full, full_domain_quadrature, load_full_model, section_readouts
from partitioned_energy_pinn import energy_residual, fit_energy_readout
from run_method_study import Tee, digest, save_json
from traction_control import EquilibriumControl, junction_scale_readout


def load_parent(checkpoint, tangent_stiffness=None):
    """Retain the saved trial space while changing the prescribed spring law."""
    model = load_full_model(checkpoint)
    if tangent_stiffness is not None:
        if not np.isfinite(tangent_stiffness) or tangent_stiffness <= 0:
            raise ValueError("Tangential stiffness must be finite and positive.")
        model.case = replace(model.case, tangent_K_MPa_per_mm=tangent_stiffness)
        model.k[1] = tangent_stiffness*model.case.specimen_size_mm/model.case.soil_E_MPa
    return model


class InteriorElimination:
    """Orthogonal elimination on energy and compact bulk-equilibrium rows."""
    def __init__(self, model, quad, control, moments, degree, rcond):
        if control.equilibrium_metric != "reference":
            raise ValueError("Interior elimination requires the common-reference equilibrium metric.")
        rectangles = [{key: row[key] for key in ("cell", "material", "bounds")}
                      for row in moments.rows if row["kind"] == "bulk_balance" and row["component"] == 0]
        self.field = MomentInteriorField(rectangles, degree)
        energy_size = energy_residual(model, quad).numel()
        bulk_weights = control.quad[1].numpy()
        bulk_ids = np.cumsum((bulk_weights > 0).reshape(-1)).reshape(bulk_weights.shape)-1
        self.blocks = []
        for rectangle in rectangles:
            cell, bounds = rectangle["cell"], rectangle["bounds"]
            operators, row_ids, counts = [], [], []
            for local_quad, length, components in ((quad, None, 3), (control.quad, control.length, 2)):
                points, weights = local_quad[0].detach().numpy(), local_quad[1].numpy()
                mask = rectangle_mask(points[cell], weights[cell], bounds)
                if not mask.any():
                    raise ValueError(f"No integration points in moment rectangle {rectangle}.")
                counts.append(int(mask.sum()))
                operators.append(local_operator(points[cell, mask], weights[cell, mask], bounds, degree,
                                                 float(model.E[cell, 0, 0]), model.case.poisson, length))
                point_ids = (cell*weights.shape[1]+np.flatnonzero(mask) if length is None
                             else bulk_ids[cell, mask])
                offset = 0 if length is None else energy_size
                row_ids.append((offset+components*point_ids[:, None]+np.arange(components)).reshape(-1))
            operator = np.vstack(operators)
            norms = np.linalg.norm(operator, axis=0)
            u, singular, vt = linalg.svd(operator/norms, full_matrices=False)
            rank = int((singular > rcond*singular[0]).sum())
            self.blocks.append({"rows": np.concatenate(row_ids), "q": u[:, :rank],
                                "inverse": (vt[:rank].T/singular[:rank])/norms[:, None],
                                "metadata": {**rectangle, "energy_points": counts[0], "bulk_points": counts[1],
                                             "columns": len(norms), "rank": rank,
                                             "retained_condition": float(singular[0]/singular[rank-1])}})

    def condense(self, matrix, base, kept_rows):
        """Replace A,b by (I-B B+) A,b and retain local recovery maps."""
        compact = np.cumsum(kept_rows)-1
        for block in self.blocks:
            rows = compact[block["rows"]]
            q = block["q"]
            block["base_response"] = q.T@base[rows]
            block["column_response"] = np.empty((q.shape[1], matrix.shape[1]))
            base[rows] -= q@block["base_response"]
            # Avoid allocating a second tall matrix for a large soil rectangle.
            for start in range(0, matrix.shape[1], 128):
                columns = slice(start, start+128)
                values = matrix[rows, columns]
                response = q.T@values
                block["column_response"][:, columns] = response
                matrix[rows, columns] = values-q@response
        print(json.dumps({"interior_blocks_eliminated": len(self.blocks),
                          "interior_columns": self.field.coefficients.numel()}), flush=True)

    def recover(self, coefficients):
        for index, block in enumerate(self.blocks):
            response = block["base_response"]+block["column_response"]@coefficients
            values = -block["inverse"]@response
            self.field.coefficients[index] = torch.tensor(values.reshape(-1, 2))
        return self.field

    def summary(self):
        return {"blocks": [block["metadata"] for block in self.blocks],
                "columns": self.field.coefficients.numel(),
                "maximum_coefficient": float(self.field.coefficients.abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/method_moments_full.json"))
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--tangent-stiffness", type=float,
                        help="Prescribed K_t in MPa/mm; retain the parent's trial space and other inputs.")
    args = parser.parse_args()
    if args.degree < 0:
        parser.error("--degree must be nonnegative")
    checkpoint, output = ROOT/args.checkpoint, ROOT/args.output
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    if config["equilibrium_control_scope"] != "full":
        parser.error("The coupled fit requires the full governing-equation objective.")
    parent = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    model = load_parent(checkpoint, args.tangent_stiffness)
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
    options = {key: value for key, value in config["equation_moments"].items() if key != "check_edge_nodes"}
    moments = EquationMoments(model, **options)
    moment_check = EquationMoments(model, **{**options, "edge_nodes": config["equation_moments"]["check_edge_nodes"]})
    neural_checkpoint = output.with_name(output.name+"_neural").with_suffix(".pt")
    trace = {"command": "python -X utf8 scripts/couple_moment_interior.py "+" ".join(sys.argv[1:]),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "parent_checkpoint_ref": args.checkpoint.as_posix(), "parent_checkpoint_hash": digest(checkpoint),
             "config_ref": args.config.as_posix(), "config_hash": digest(ROOT/args.config),
             "case": asdict(model.case),
             "transcript_ref": output.with_suffix(".log").relative_to(ROOT).as_posix(),
             "neural_checkpoint_ref": neural_checkpoint.relative_to(ROOT).as_posix()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(json.dumps({"starting": output.name}), flush=True)
        before = objective_readout(model, quad, training)
        elimination = InteriorElimination(model, quad, training, moments, args.degree, settings["rcond"])
        fit = fit_energy_readout(model, quad, settings["rcond"], training, moments, elimination)
        # This checkpoint intentionally holds only neural coefficients; the
        # complete displacement is recovered from the JSON field recipe.
        model.save(neural_checkpoint)
        attach_interior(model, elimination.field)
        after = objective_readout(model, quad, training)
        checked = evaluate_full(model, settings["check_nodes"], settings["check_edge_nodes"],
                                settings["region_nodes"], settings["check_support_nodes"])
        finer_sections = section_readouts(model, checked["observation_supports"]["section_y"], 384)
        scale = checked["scales"]["line_resultant_N_per_mm_thickness"]
        verification = {
            "probe_moment_max_N_per_mm": float(moments(elimination.field).abs().max())*scale,
            "objective_reconstruction_difference": abs(after["objective"]-fit["objective_after"]),
            "section_256_to_384_max_change_N_per_mm": max(
                abs(a["total_force"]-b["total_force"])*scale for a, b in zip(checked["sections"], finer_sections))}
        payload = {"kind": "coupled_moment_interior_fit", "execution": trace,
                   "question": ("How does the prescribed tangential stiffness change transfer and load sharing in the same coupled trial space?"
                                if args.tangent_stiffness is not None else
                                "How does coupling the tested interior variations to the neural traces change the observed transfer and load path?"),
                   "optimization": fit, "interior_field": elimination.field.export(),
                   "training_objective": {"before": before, "after": after},
                   "independent_control": independent.readout(),
                   "equation_moments": {"training": moments.readout(), "independent": moment_check.readout()},
                   "parent_independent_readout": parent["independent_readout"], "independent_readout": checked,
                   "neighboring_scale_readout": junction_scale_readout(model, [.375, .75], settings["check_edge_nodes"]),
                   "verification": verification}
        save_json(output.with_suffix(".json"), payload)
        replay = load_probe_field(output.with_suffix(".json"))
        replay_sections = section_readouts(replay, checked["observation_supports"]["section_y"], settings["check_edge_nodes"])
        verification["saved_field_section_replay_max_difference_N_per_mm"] = max(
            abs(a["total_force"]-b["total_force"])*scale for a, b in zip(checked["sections"], replay_sections))
        trace.update({"finished_at": datetime.now(timezone.utc).isoformat(), "exit_code": 0,
                      "neural_checkpoint_hash": digest(neural_checkpoint)})
        save_json(output.with_suffix(".json"), payload)
        print(json.dumps({"completed": output.name, "verification": verification,
                          "nominal_stiffness_MPa": checked["nominal_axial_stiffness_MPa"],
                          "section_forces_N_per_mm": [r["total_force"]*scale for r in checked["sections"]]}), flush=True)


if __name__ == "__main__":
    main()
