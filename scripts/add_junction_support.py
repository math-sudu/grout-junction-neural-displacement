"""Refit a saved full-domain field, optionally adding junction variation."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import torch

from full_domain_pinn import (CellInteriorModes, IndependentLocalModes, InterfaceLocalModes, evaluate_full,
                              full_domain_quadrature, full_probe_work, load_full_model)
from partitioned_energy_pinn import fit_energy_readout
from partitioned_mixed_pinn import ROOT
from run_method_study import Tee, digest, export_fields, save_json


def add_support(model, radius_factor, width, seed):
    if model.additional_local is not None:
        raise ValueError("The checkpoint already contains an additional local support.")
    settings = {"radius": radius_factor*model.vein_width, "width": width, "seed": seed}
    model.additional_local = IndependentLocalModes(model.case, model.geometry, **settings)
    model.options["additional_local"] = settings


def split_support_interior(model):
    local = model.additional_local
    if local is None or local.interior_coefficients is not None:
        raise ValueError("Interior splitting requires an existing unsplit additional support.")
    local.interior_coefficients = torch.nn.Parameter(torch.zeros_like(local.coefficients))
    model.options["additional_local"]["split_interior"] = True


def release_cell_interiors(model):
    """Copy each material readout to its cells, retaining the parent field."""
    local = model.additional_local
    if local is None or local.interior_coefficients is None or local.cellwise_interior:
        raise ValueError("Cellwise release requires existing material-shared interior readouts.")
    local.interior_coefficients = torch.nn.Parameter(
        local.interior_coefficients.detach()[:, local.material_ids].clone())
    local.cellwise_interior = True
    model.options["additional_local"]["cellwise_interior"] = True


def add_interface_traces(model, radius_factor, width, seed):
    if model.interface_local is not None:
        raise ValueError("The checkpoint already contains independent interface-local traces.")
    settings = {"radius": radius_factor*model.vein_width, "width": width, "seed": seed}
    model.interface_local = InterfaceLocalModes(model.case, model.geometry, **settings)
    model.options["interface_local"] = settings


def add_cell_interiors(model, radius_factor, width, seed):
    if model.cell_interior is not None:
        raise ValueError("The checkpoint already contains compact cell-interior fields.")
    settings = {"radius": radius_factor*model.vein_width, "width": width, "seed": seed}
    model.cell_interior = CellInteriorModes(model.case, model.geometry, **settings)
    model.options["cell_interior"] = settings


def add_shared_traces(model, radius_factor, width, seed):
    if model.shared_local is not None:
        raise ValueError("The checkpoint already contains compact shared traces.")
    settings = {"radius": radius_factor*model.vein_width, "width": width, "seed": seed}
    model.shared_local = InterfaceLocalModes(model.case, model.geometry, shared=True, **settings)
    model.options["shared_local"] = settings


def add_junction_patches(model, radius_factors, width, seed):
    settings = [{"radius": factor*model.vein_width, "width": width, "seed": seed+i}
                for i, factor in enumerate(radius_factors)]
    for entry in settings:
        model.junction_patches.append(model.make_junction_patch(entry))
    model.options["junction_patches"] = (model.options["junction_patches"] or [])+settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius-factor", type=float, default=2.)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--seed", type=int, default=29)
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument("--split-existing-interior", action="store_true",
                              help="Retain the broad field and add trace-free cell-interior readouts using its hidden features.")
    intervention.add_argument("--refit-only", action="store_true",
                              help="Keep the saved representation and refit coefficients with the supplied integration settings.")
    intervention.add_argument("--cellwise-existing-interior", action="store_true",
                              help="Release material-shared interior readouts into independent cell readouts, retaining the parent field.")
    intervention.add_argument("--interface-traces", action="store_true",
                              help="Add independent compact neural traces on both sides of each junction interface.")
    intervention.add_argument("--cell-interiors", action="store_true",
                              help="Add compact zero-trace neural fields with independent junction/cell readouts.")
    intervention.add_argument("--shared-traces", action="store_true",
                              help="Add compact neural traces shared across each same-material junction cut.")
    intervention.add_argument("--refine-junction", nargs="+", type=float, metavar="RADIUS_FACTOR",
                              help="Add interface, shared-cut and cell-interior fields at each radius relative to arm width.")
    parser.add_argument("--config", type=Path, default=Path("config/method_study.json"))
    args = parser.parse_args()
    checkpoint, output = ROOT/args.checkpoint, ROOT/args.output
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    settings = config["full_domain"]
    parent = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    model = load_full_model(checkpoint)
    if args.split_existing_interior:
        split_support_interior(model)
    elif args.cellwise_existing_interior:
        release_cell_interiors(model)
    elif args.interface_traces:
        add_interface_traces(model, args.radius_factor, args.width, args.seed)
    elif args.cell_interiors:
        add_cell_interiors(model, args.radius_factor, args.width, args.seed)
    elif args.shared_traces:
        add_shared_traces(model, args.radius_factor, args.width, args.seed)
    elif args.refine_junction:
        add_junction_patches(model, args.refine_junction, args.width, args.seed)
    elif not args.refit_only:
        add_support(model, args.radius_factor, args.width, args.seed)
    trace = {"command": "python -X utf8 scripts/add_junction_support.py "+" ".join(sys.argv[1:]),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "config_ref": args.config.as_posix(), "config_hash": digest(ROOT/args.config),
             "parent_checkpoint_ref": args.checkpoint.as_posix(), "parent_checkpoint_hash": digest(checkpoint),
             "additional_local": model.options["additional_local"],
             "interface_local": model.options["interface_local"],
             "cell_interior": model.options["cell_interior"],
             "shared_local": model.options["shared_local"],
             "junction_patches": model.options["junction_patches"],
             "intervention": "integration_refit" if args.refit_only else
             "junction_refinement" if args.refine_junction else
             "shared_traces" if args.shared_traces else
             "cell_interiors" if args.cell_interiors else
             "interface_traces" if args.interface_traces else
             "cellwise_interior" if args.cellwise_existing_interior else
             "split_interior" if args.split_existing_interior else "add_support",
             "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
             "transcript_ref": output.with_suffix(".log").relative_to(ROOT).as_posix()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(json.dumps({"starting": output.name}), flush=True)
        quad = list(full_domain_quadrature(model, settings["nodes"], settings["edge_nodes"], settings.get("support_nodes")))
        quad[0].requires_grad_(True)
        fit = fit_energy_readout(model, quad, settings["rcond"])
        model.save(output.with_suffix(".pt"))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trained = evaluate_full(model, settings["nodes"], settings["edge_nodes"], settings["region_nodes"],
                                settings.get("support_nodes"))
        checked = evaluate_full(model, settings["check_nodes"], settings["check_edge_nodes"], settings["region_nodes"],
                                settings.get("check_support_nodes"))
        probes = full_probe_work(model, settings["check_nodes"], settings["check_edge_nodes"],
                                 model.additional_local.radius if model.additional_local is not None else None,
                                 settings.get("check_support_nodes"))
        export_fields(model, output.with_suffix(".csv"), settings["check_nodes"])
        trace.update({"exit_code": 0, "finished_at": datetime.now(timezone.utc).isoformat(),
                      "run_output_paths": [output.with_suffix(suffix).relative_to(ROOT).as_posix()
                                           for suffix in (".json", ".pt", ".csv")]})
        save_json(output.with_suffix(".json"), {
            "run": {**parent["run"], "id": output.name,
                    "representation": parent["run"]["representation"] if args.refit_only else
                    "local_multiscale_junction" if args.refine_junction else
                    "local_shared_traces" if args.shared_traces else
                    "local_narrow_interiors" if args.cell_interiors else
                    "local_interface_traces" if args.interface_traces else
                    "local_cellwise_interior" if args.cellwise_existing_interior else
                    "local_split_interior" if args.split_existing_interior else "local_additive_support"},
            "interface_state": parent["interface_state"], "options": model.options,
            "execution": trace, "optimization": fit, "hidden_feature_updates": 0,
            "training_readout": trained, "independent_readout": checked, "virtual_work": probes})
        print(json.dumps({"completed": output.name, "nominal_stiffness_MPa": checked["nominal_axial_stiffness_MPa"]}), flush=True)


if __name__ == "__main__":
    main()
