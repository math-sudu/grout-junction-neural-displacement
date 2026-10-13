"""Execute the prescribed cross and offset-T representation interventions.

All ordinary runs solve readout coefficients with hidden features fixed.
Hidden-feature adaptation is a separate, available intervention, chosen only
when the target observables and virtual-work diagnosis motivate it.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

from full_domain_pinn import FullDomainCase, FullDomainNet, evaluate_full, full_probe_work
from local_neural_enrichment import LocalNeuralModes
from local_virtual_work import probe_work
from partitioned_energy_pinn import (energy_response, fit_energy_readout,
                                     junction_resultants, load_model)
from partitioned_mixed_pinn import ROOT, quadrature


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def input_fingerprint(config):
    refs = [config[key] for key in ("input_ref", "source_input_ref", "cross_checkpoint_ref")]
    hashes = []
    for ref in refs:
        if ref.endswith(".json"):
            # Input content is independent of Git's Windows newline conversion.
            content = json.dumps(json.loads((ROOT/ref).read_text(encoding="utf-8")),
                                 sort_keys=True, separators=(",", ":")).encode()
            value = hashlib.sha256(content).hexdigest()
        else:
            value = digest(ROOT/ref)
        hashes.append(f"{ref}:{value}")
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8", newline="\n")


def export_fields(model, path, nodes):
    xy = quadrature(model.cells, nodes, 2)[0].requires_grad_(True)
    uv, strain = model.displacement_strain(xy)
    stress = model.stress(strain)
    values = torch.cat([xy, uv, stress, strain], -1).detach().numpy()
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["cell", "material", "x_over_H", "y_over_H", "ux_over_Delta", "uy_over_Delta",
                         "sxx", "syy", "sxy", "exx", "eyy", "gamma_xy"])
        for index, (cell, rows) in enumerate(zip(model.cells, values)):
            writer.writerows([index, cell["material"], *row] for row in rows)


def run_full(config, run, output):
    inputs = json.loads((ROOT/config["input_ref"]).read_text(encoding="utf-8"))
    interface = next(row for row in inputs["interface_states"] if row["id"] == run["interface"])
    case = FullDomainCase(**{**inputs["case"], "normal_K_MPa_per_mm": interface["K_n_MPa_per_mm"],
                             "tangent_K_MPa_per_mm": interface["K_t_MPa_per_mm"]})
    settings = config["full_domain"]
    geometry_spec = {key: inputs["geometry"][key] for key in
                     ("geometry_id", "bounds", "rectangles", "arm_width", "junction_location", "branch_endpoints")}
    model = FullDomainNet(case, geometry_spec, settings["width"], settings["edge_width"], config["seed"],
                          run["representation"] != "base", settings["local"] if run["representation"] == "local" else None)
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
    save_json(output.with_suffix(".json"), {"run": run, "interface_state": interface, "options": model.options,
                                           "optimization": fit, "hidden_feature_updates": 0,
                                           "training_readout": trained, "independent_readout": checked,
                                           "virtual_work": probes})
    return {"id": run["id"], "energy": checked["stored_energy"],
            "nominal_axial_stiffness_MPa": checked["nominal_axial_stiffness_MPa"]}


def run_cross(config, run, output):
    model, seed = load_model(ROOT/config["cross_checkpoint_ref"])
    settings = config["cross_support"]
    additional = {"radius": settings["radius_factor"]*model.vein_width,
                  "width": settings["width"], "seed": settings["seed"]}
    model.additional_local = LocalNeuralModes(model.case, model.cells, model.vein_width/2, **additional)
    quad = list(quadrature(model.cells, settings["nodes"], settings["edge_nodes"]))
    quad[0].requires_grad_(True)
    fit = fit_energy_readout(model, quad, settings["rcond"])
    metadata = {"case": asdict(model.case), "width": model.width, "seed": seed,
                "kinematics": model.kinematics, "displacement_scaling": model.displacement_scaling,
                "corner_enrichment": model.corner_enrichment, "elastic_corners": model.elastic_corners,
                "junction_radius": model.junction.radius, "local_neural": model.local.metadata(),
                "additional_local": additional, "formulation": "displacement_energy"}
    torch.save({**metadata, "state_dict": model.state_dict()}, output.with_suffix(".pt"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    response, fields, edges, sections = energy_response(model, settings["check_nodes"],
                                                        settings["check_edge_nodes"], settings["region_nodes"])
    fields.to_csv(output.with_suffix(".csv"), index=False)
    edges.to_csv(output.with_name(output.name+"_interfaces.csv"), index=False)
    sections.to_csv(output.with_name(output.name+"_sections.csv"), index=False)
    payload = {"run": run, **metadata, "parent_checkpoint": config["cross_checkpoint_ref"],
               "optimization": fit, "hidden_feature_updates": 0, "independent_readout": response,
               "junction_transfer": {"nodes_per_segment": 128, "segments": junction_resultants(model, 128)},
               "virtual_work": [probe_work(model, settings["check_nodes"], settings["check_edge_nodes"], factor*model.vein_width)
                                for factor in (1., 2.)]}
    save_json(output.with_suffix(".json"), payload)
    return {"id": run["id"], "energy": response["stored_energy"],
            "nominal_axial_stiffness_MPa": response["axial_modulus_MPa"]}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/method_study.json"))
    parser.add_argument("--run", action="append", help="Execute a named run only; repeat to select several.")
    args = parser.parse_args()
    config_path = ROOT/args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = config["runs"]
    if args.run:
        unknown = set(args.run)-{row["id"] for row in runs}
        if unknown:
            parser.error(f"Unknown run IDs: {sorted(unknown)}")
        runs = [row for row in runs if row["id"] in args.run]
    directory = ROOT/config["output_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    trace = {"command": "python -X utf8 scripts/run_method_study.py "+" ".join(sys.argv[1:]),
             "config_hash": digest(config_path), "data_fingerprint": input_fingerprint(config),
             "started_at": datetime.now(timezone.utc).isoformat(), "run_ids": [r["id"] for r in runs]}
    label = "study" if args.run is None else "+".join(args.run)
    transcript = directory/(label+".log")
    trace["transcript_ref"] = transcript.relative_to(ROOT).as_posix()
    summaries = []
    with transcript.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)):
        for run in runs:
            print(json.dumps({"starting": run["id"]}), flush=True)
            output = directory/run["id"]
            execute = run_cross if run["representation"] == "cross_additive_support" else run_full
            summaries.append(execute(config, run, output))
            print(json.dumps({"completed": summaries[-1]}), flush=True)
    trace.update({"exit_code": 0, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "run_output_paths": [str((directory/(r["id"]+suffix)).relative_to(ROOT).as_posix())
                                       for r in runs for suffix in (".json", ".pt", ".csv")]})
    save_json(directory/(label+".json"), {"execution": trace, "summaries": summaries})


if __name__ == "__main__":
    main()
