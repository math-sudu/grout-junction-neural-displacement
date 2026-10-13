"""Refit a compatible displacement checkpoint with equilibrium/traction control."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from partitioned_mixed_pinn import ROOT

import torch

from full_domain_pinn import evaluate_full, full_domain_quadrature, load_full_model
from partitioned_energy_pinn import fit_energy_readout
from run_method_study import Tee, digest, export_fields, save_json
from traction_control import EquilibriumControl, junction_balance_readout, junction_scale_readout
from equation_moments import EquationMoments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/method_moments_full.json"))
    args = parser.parse_args()
    checkpoint, output = ROOT/args.checkpoint, ROOT/args.output
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    parent = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    model = load_full_model(checkpoint)
    settings, control = config["full_domain"], config["equilibrium_control"]
    length = model.vein_width*control["length_over_arm_width"]
    metric = control.get("interface_metric", "material")
    equilibrium_metric = control.get("equilibrium_metric", "material")
    training = EquilibriumControl(model, control["nodes"], control["edge_nodes"],
                                  control["support_nodes"], length, metric, equilibrium_metric)
    independent = EquilibriumControl(model, control["check_nodes"], control["check_edge_nodes"],
                                     control["check_support_nodes"], length, metric, equilibrium_metric)
    moment_settings = config.get("equation_moments")
    moments = moment_check = None
    if moment_settings:
        options = {key: value for key, value in moment_settings.items() if key != "check_edge_nodes"}
        moments = EquationMoments(model, **options)
        moment_check = EquationMoments(model, **{**options, "edge_nodes": moment_settings["check_edge_nodes"]})
    # Retain the objective of saved run configurations that predate this selector.
    scope = config.get("equilibrium_control_scope",
                       "bulk" if config.get("bulk_equilibrium_control", False)
                       else "none" if moments else "full")
    if scope not in ("none", "bulk", "full"):
        raise ValueError(f"Unknown equilibrium control scope: {scope}")
    residual = {"none": None, "bulk": training.bulk_residual, "full": training}[scope]
    if moments:
        intervention, objective = {
            "none": ("equation_moment_constraints", "energy_with_equation_moments"),
            "bulk": ("equation_moments_with_bulk_equilibrium", "energy_plus_bulk_with_equation_moments"),
            "full": ("equation_moments_with_full_equilibrium", "energy_plus_full_equilibrium_with_equation_moments"),
        }[scope]
    else:
        intervention, objective = {
            "none": ("energy_readout", "energy"),
            "bulk": ("consistent_bulk_equilibrium_control", "energy_plus_bulk_equilibrium"),
            "full": ("consistent_equilibrium_traction_control", "energy_plus_equilibrium_traction"),
        }[scope]
    trace = {"command": "python -X utf8 scripts/refit_traction_control.py "+" ".join(sys.argv[1:]),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "config_ref": args.config.as_posix(), "config_hash": digest(ROOT/args.config),
             "parent_checkpoint_ref": args.checkpoint.as_posix(), "parent_checkpoint_hash": digest(checkpoint),
             "intervention": intervention,
             "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
             "torch_num_threads": torch.get_num_threads(),
             "transcript_ref": output.with_suffix(".log").relative_to(ROOT).as_posix()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(json.dumps({"starting": output.name}), flush=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        before = {"training": training.readout(), "independent": independent.readout()}
        moment_before = ({"training": moments.readout(), "independent": moment_check.readout()}
                         if moments else None)
        # Read scales on either side of the constrained half-arm-width patch.
        prediction_radii = [.375, .75] if moments else None
        prediction_before = (junction_scale_readout(model, prediction_radii, settings["check_edge_nodes"])
                             if prediction_radii else None)
        before_balance = junction_balance_readout(model, settings["check_edge_nodes"])
        print(json.dumps({"parent_control": before}), flush=True)
        quad = list(full_domain_quadrature(model, settings["nodes"], settings["edge_nodes"], settings["support_nodes"]))
        quad[0].requires_grad_(True)
        fit = fit_energy_readout(model, quad, settings["rcond"],
                                 residual, moments)
        model.save(output.with_suffix(".pt"))
        trained = evaluate_full(model, settings["nodes"], settings["edge_nodes"], settings["region_nodes"],
                                settings["support_nodes"])
        checked = evaluate_full(model, settings["check_nodes"], settings["check_edge_nodes"], settings["region_nodes"],
                                settings["check_support_nodes"])
        after = {"training": training.readout(), "independent": independent.readout()}
        export_fields(model, output.with_suffix(".csv"), settings["check_nodes"])
        trace.update({"exit_code": 0, "finished_at": datetime.now(timezone.utc).isoformat(),
                      "run_output_paths": [output.with_suffix(s).relative_to(ROOT).as_posix()
                                           for s in (".json", ".pt", ".csv")]})
        payload = {
            "run": {**parent["run"], "id": output.name, "objective": objective},
            "interface_state": parent["interface_state"], "options": model.options,
            "execution": trace, "optimization": fit, "hidden_feature_updates": 0,
            "equilibrium_control": {"settings": control, "scope": scope, "before": before, "after": after},
            "junction_balance": {"before": before_balance,
                                 "after": junction_balance_readout(model, settings["check_edge_nodes"])},
            "training_readout": trained, "independent_readout": checked}
        if moments:
            payload["equation_moments"] = {"settings": moment_settings, "before": moment_before,
                                            "after": {"training": moments.readout(),
                                                      "independent": moment_check.readout()}}
        if prediction_radii:
            payload["neighboring_scale_readout"] = {
                "before": prediction_before,
                "after": junction_scale_readout(model, prediction_radii, settings["check_edge_nodes"])}
        save_json(output.with_suffix(".json"), payload)
        print(json.dumps({"completed": output.name, "nominal_stiffness_MPa": checked["nominal_axial_stiffness_MPa"],
                          "control_after": after}), flush=True)


if __name__ == "__main__":
    main()
