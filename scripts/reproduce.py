"""Generate the symmetric-cross and offset-T fields from prescribed inputs.

The numerical kernels and fitting routines are the paper's existing solvers.
The T assembly omits intermediate linear fits: the final solver zeros all
readouts and reconstructs the full affine least-squares problem. Hidden-feature
adaptation and every support's initialization are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CROSS_DIR = "results/feasibility/partitioned_2d"
T_DIR = "results/method_study"
STUDY_CONFIG = "config/method_study.json"
FINAL_CONFIG = "config/method_moments_full.json"


def read_json(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def prepare_inputs():
    from partitioned_mixed_pinn import Case
    from prepare_method_inputs import main as prepare_t

    case = read_json("config/problem.json")
    if asdict(Case(**case)) != asdict(Case(geometry="cross")):
        raise ValueError("The historical cross training recipe uses the prescribed Case defaults; "
                         "update Case and the recipe together when changing this problem.")
    directory = ROOT/"data/inputs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"forward_inputs.json").write_text(
        json.dumps({"case": case}, indent=2)+"\n", encoding="utf-8")
    prepare_t()


def run_cross_support():
    from run_method_study import run_cross

    config = read_json(STUDY_CONFIG)
    output = ROOT/T_DIR/"cross_additive_support"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_cross(config, {"id": output.name, "representation": "cross_additive_support"}, output)


def run_t_initial():
    from run_method_study import run_full

    config = read_json(STUDY_CONFIG)
    directory = ROOT/T_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for run in config["runs"]:
        if run["id"].startswith("t_"):
            print(json.dumps({"starting": run["id"]}), flush=True)
            run_full(config, run, directory/run["id"])


def assemble_t_space(model):
    """Append the final zero-readout fields to the adapted narrow-local field."""
    from add_junction_support import (
        add_cell_interiors, add_interface_traces, add_junction_patches,
        add_shared_traces, add_support, release_cell_interiors,
        split_support_interior,
    )

    add_support(model, 2., 12, 29)
    split_support_interior(model)
    release_cell_interiors(model)
    add_interface_traces(model, .5, 6, 43)
    add_cell_interiors(model, .5, 6, 47)
    add_shared_traces(model, .5, 6, 53)
    add_junction_patches(model, [.1, .2], 4, 59)
    return model


def prepare_t_space():
    from full_domain_pinn import load_full_model
    from run_method_study import save_json

    parent = ROOT/T_DIR/"t_equal_adapted.pt"
    model = assemble_t_space(load_full_model(parent))
    output = ROOT/T_DIR/"t_final_trial_space"
    model.save(output.with_suffix(".pt"))
    metadata = read_json(parent.with_suffix(".json"))
    # All appended readouts are zero, so this readout is also the prepared
    # field's readout. The final coupled solve refits every coefficient.
    save_json(output.with_suffix(".json"), {
        "run": {"id": output.name, "representation": "final_trial_space"},
        "options": model.options,
        "independent_readout": metadata["independent_readout"],
    })
    print(json.dumps({"prepared": str(output.relative_to(ROOT)),
                      "neural_coefficients": sum(p.numel() for p in model.readout_parameters())}),
          flush=True)


def export_t_fields():
    from diagnose_moment_interior import load_probe_field
    from run_method_study import export_fields

    for name in ("t_equal_moment_coupled", "t_sliding_moment_coupled"):
        path = ROOT/T_DIR/name
        model = load_probe_field(path.with_suffix(".json"))
        export_fields(model, path.with_suffix(".csv"), 64)
        print(json.dumps({"complete_field_csv": str(path.with_suffix(".csv").relative_to(ROOT))}), flush=True)


def command(script, *arguments):
    return [sys.executable, "-X", "utf8", "-u", f"scripts/{script}.py", *map(str, arguments)]


def recipe(case):
    steps = [("inputs", command("reproduce", "--stage", "inputs"))]
    if case in ("all", "cross"):
        mixed = [
            ("cross_shared", None, ["--geometry", "cross", "--steps", 6500,
                                   "--displacement-scaling", "unit"]),
            ("cross_scaled", "cross_shared", ["--steps", 4000,
                                               "--displacement-scaling", "compliance"]),
            ("cross_quadrature", "cross_scaled", ["--steps", 2200, "--nodes", 20, "--edge-nodes", 48]),
            ("cross_augmented", "cross_quadrature", ["--steps", 600, "--nodes", 20,
                                                       "--edge-nodes", 48, "--dual-updates", 6]),
            ("cross_augmented_continued", "cross_augmented", ["--steps", 800, "--nodes", 20,
                                                              "--edge-nodes", 48, "--dual-updates", 4]),
            ("cross_augmented_svd_integrated", "cross_augmented_continued",
             ["--steps", 0, "--nodes", 40, "--edge-nodes", 48, "--dual-updates", 6,
              "--readout-solver", "svd", "--readout-rcond", 1e-9]),
        ]
        for name, parent, arguments in mixed:
            if parent:
                arguments += ["--resume", f"{CROSS_DIR}/{parent}.pt"]
            steps.append((name, command("partitioned_mixed_pinn", "--name", name, *arguments)))
        energy = [
            ("cross_energy_fixed", "cross_augmented_svd_integrated", []),
            ("cross_energy_log", "cross_energy_fixed", ["--add-elastic-corners"]),
            ("cross_energy_combined", "cross_energy_log", ["--junction-radius", .15100040032032047]),
            ("cross_energy_local", "cross_energy_combined", ["--local-radius", .0755002001601602,
                                                               "--local-width", 12]),
            ("cross_energy_local_adapted", "cross_energy_local", ["--local-steps", 400]),
        ]
        for name, parent, arguments in energy:
            steps.append((name, command("partitioned_energy_pinn", "--name", name,
                                         "--resume", f"{CROSS_DIR}/{parent}.pt",
                                         "--rcond", 1e-9, *arguments)))
        steps.append(("cross_additive_support", command("reproduce", "--stage", "cross-support")))
    if case in ("all", "offset-t"):
        steps += [
            ("t_initial", command("reproduce", "--stage", "t-initial")),
            ("t_equal_adapted", command("refine_local_features", f"{T_DIR}/t_equal_local.pt",
                                         "--output", f"{T_DIR}/t_equal_adapted", "--steps", 200,
                                         "--config", STUDY_CONFIG)),
            ("t_final_trial_space", command("reproduce", "--stage", "t-space")),
            ("t_equal_moment_coupled", command("couple_moment_interior", f"{T_DIR}/t_final_trial_space.pt",
                                               "--output", f"{T_DIR}/t_equal_moment_coupled",
                                               "--config", FINAL_CONFIG)),
            ("t_sliding_moment_coupled", command("couple_moment_interior", f"{T_DIR}/t_final_trial_space.pt",
                                                 "--output", f"{T_DIR}/t_sliding_moment_coupled",
                                                 "--tangent-stiffness", 1, "--config", FINAL_CONFIG)),
            ("t_export", command("reproduce", "--stage", "t-export")),
        ]
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "cross", "offset-t"), default="all")
    parser.add_argument("--list", action="store_true", help="Print the exact ordered commands without executing them.")
    parser.add_argument("--start-at", help="Resume at a listed stage; earlier generated files must already exist.")
    parser.add_argument("--stage", choices=("inputs", "cross-support", "t-initial", "t-space", "t-export"),
                        help="Run one preparation/export stage used by the recipe.")
    args = parser.parse_args()
    if args.stage:
        actions = {"inputs": prepare_inputs, "cross-support": run_cross_support,
                   "t-initial": run_t_initial, "t-space": prepare_t_space, "t-export": export_t_fields}
        actions[args.stage]()
        return
    steps = recipe(args.case)
    if args.start_at:
        names = [name for name, _ in steps]
        if args.start_at not in names:
            parser.error(f"Unknown stage {args.start_at!r}; use --list to inspect this recipe.")
        steps = steps[names.index(args.start_at):]
    environment = os.environ.copy()
    environment.setdefault("OPENBLAS_NUM_THREADS", "2")
    for name, argv in steps:
        display = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        print(f"[{name}] {display}", flush=True)
        if not args.list:
            subprocess.run(argv, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
