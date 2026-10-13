"""Test hidden-feature adaptation on a saved full-domain local field.

The inherited field is fixed during local adaptation; a subsequent linear
coefficient solve restores stationarity in the complete current trial space.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from full_domain_pinn import evaluate_full, full_probe_work, load_full_model
from local_neural_enrichment import adapt_local_features
from partitioned_energy_pinn import fit_energy_readout
from partitioned_mixed_pinn import ROOT, quadrature
from run_method_study import Tee, digest, export_fields, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--config", type=Path, default=Path("config/method_study.json"))
    args = parser.parse_args()
    checkpoint, output = ROOT / args.checkpoint, ROOT / args.output
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    settings = config["full_domain"]
    parent = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    model = load_full_model(checkpoint)
    trace = {"command": "python -X utf8 scripts/refine_local_features.py " + " ".join(sys.argv[1:]),
             "started_at": datetime.now(timezone.utc).isoformat(),
             "config_ref": args.config.as_posix(), "config_hash": digest(ROOT / args.config),
             "parent_checkpoint_ref": args.checkpoint.as_posix(), "parent_checkpoint_hash": digest(checkpoint),
             "maximum_adaptation_iterations": args.steps,
             "transcript_ref": output.with_suffix(".log").relative_to(ROOT).as_posix()}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        print(json.dumps({"starting": output.name}), flush=True)
        adaptation = adapt_local_features(model, settings["nodes"], settings["edge_nodes"], args.steps)
        print(json.dumps({"adaptation": adaptation}), flush=True)
        quad = list(quadrature(model.cells, settings["nodes"], settings["edge_nodes"]))
        quad[0].requires_grad_(True)
        fit = fit_energy_readout(model, quad, settings["rcond"])
        model.save(output.with_suffix(".pt"))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trained = evaluate_full(model, settings["nodes"], settings["edge_nodes"], settings["region_nodes"])
        checked = evaluate_full(model, settings["check_nodes"], settings["check_edge_nodes"], settings["region_nodes"])
        probes = full_probe_work(model, settings["check_nodes"], settings["check_edge_nodes"])
        export_fields(model, output.with_suffix(".csv"), settings["check_nodes"])
        trace.update({"exit_code": 0, "finished_at": datetime.now(timezone.utc).isoformat(),
                      "run_output_paths": [output.with_suffix(suffix).relative_to(ROOT).as_posix()
                                           for suffix in (".json", ".pt", ".csv")]})
        save_json(output.with_suffix(".json"), {
            "run": {**parent["run"], "id": output.name, "representation": "local_adapted"},
            "interface_state": parent["interface_state"], "options": model.options,
            "execution": trace, "adaptation": adaptation, "optimization": fit,
            "hidden_feature_updates": adaptation["iterations"],
            "training_readout": trained, "independent_readout": checked, "virtual_work": probes})
        print(json.dumps({"completed": output.name, "nominal_stiffness_MPa": checked["nominal_axial_stiffness_MPa"]}), flush=True)


if __name__ == "__main__":
    main()
