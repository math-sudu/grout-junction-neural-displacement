"""Materialize the prescribed geometry, finite interface states and supports."""
import json

from full_domain_geometry import build_geometry, observation_supports
from partitioned_mixed_pinn import ROOT


def main():
    inherited = json.loads((ROOT/"data/inputs/forward_inputs.json").read_text(encoding="utf-8"))
    geometry = build_geometry()
    case = {key: value for key, value in inherited["case"].items()
            if key not in ("geometry", "grout_area_fraction", "vertical_span", "horizontal_span")}
    case["geometry"] = "offset_t"
    height = case["specimen_size_mm"]
    length = geometry["arm_width"]*height
    states = []
    for label, kn, kt in [("equal", 10., 10.), ("sliding", 10., 1.)]:
        states.append({"id": label, "K_n_MPa_per_mm": kn, "K_t_MPa_per_mm": kt,
                       "E_s_MPa": case["soil_E_MPa"], "L_mm": length, "L_role": "arm_width",
                       "K_n_L_over_E_s": kn*length/case["soil_E_MPa"],
                       "K_t_L_over_E_s": kt*length/case["soil_E_MPa"],
                       "normal_interface_length_mm": case["soil_E_MPa"]/kn,
                       "tangent_interface_length_mm": case["soil_E_MPa"]/kt})
    payload = {
        "kind": "prescribed_method_inputs", "input_role": "Forward method study; prescribed geometry and interface stiffnesses",
        "source_input_ref": "data/inputs/forward_inputs.json", "case": case,
        "geometry": geometry, "observation_supports": observation_supports(geometry),
        "physical_geometry": {
            "specimen_bounds_mm": [v*height for v in geometry["bounds"]],
            "grout_rectangles_mm": [[v*height for v in r] for r in geometry["rectangles"]],
            "junction_location_mm": [v*height for v in geometry["junction_location"]],
            "stem_width_mm": length, "cap_width_mm": length,
            "branch_endpoints_mm": [[v*height for v in p] for p in geometry["branch_endpoints"]],
            "grout_union_area_mm2": geometry["grout_area"]*height**2},
        "interface_states": states,
        "observation_units": {"coordinates": "H", "line_integrals": "E_s*Delta N/mm thickness",
                              "normal_tangent_jump": "Delta", "soil_mean_compression": "Delta/H",
                              "sections": "Full width; keep grout and total force beside their ratio"},
        "boundary_conditions": "Top u_y=-Delta/2; bottom u_y=+Delta/2; tangential platen traction and both lateral tractions zero. Fix u_x at the bottom-left soil vertex only.",
        "reproduction_command": "python -X utf8 scripts/prepare_method_inputs.py",
    }
    path = ROOT/"data/inputs/offset_t_inputs.json"
    path.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": path.relative_to(ROOT).as_posix(), "cells": len(geometry["cells"]),
                      "corners": len(geometry["corners"]), "area_mm2": payload["physical_geometry"]["grout_union_area_mm2"]}))


if __name__ == "__main__":
    main()
