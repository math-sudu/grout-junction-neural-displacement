"""Conforming cells and oriented material corners for a prescribed offset T.

Coordinates are normalized by specimen height. The full square is [-.5,.5]^2.
The cap and stem meet along a same-material edge, with no overlap in area.
"""
from __future__ import annotations

import numpy as np


OFFSET_T = {
    "geometry_id": "offset_t",
    "bounds": [-.5, .5, -.5, .5],
    "rectangles": [[-.28, .32, .16, .26], [.04, .14, -.24, .16]],
    "arm_width": .1,
    "junction_location": [.09, .16],
    "branch_endpoints": [[-.28, .21], [.32, .21], [.09, -.24]],
}


def material_at(point, rectangles):
    x, y = point
    return "grout" if any(x0 <= x <= x1 and y0 <= y <= y1
                          for x0, x1, y0, y1 in rectangles) else "soil"


def build_geometry(spec=OFFSET_T):
    bounds, rectangles = spec["bounds"], spec["rectangles"]
    x0, x1, y0, y1 = bounds
    for a, b, c, d in rectangles:
        if not x0 < a < b < x1 or not y0 < c < d < y1:
            raise ValueError("Grout rectangles must be embedded inside the specimen.")
    xs = sorted({x0, x1, *(v for r in rectangles for v in r[:2])})
    ys = sorted({y0, y1, *(v for r in rectangles for v in r[2:])})
    nx, ny = len(xs)-1, len(ys)-1
    cells = [{"bounds": [a, b, c, d],
              "material": material_at(((a+b)/2, (c+d)/2), rectangles)}
             for c, d in zip(ys[:-1], ys[1:]) for a, b in zip(xs[:-1], xs[1:])]
    edges, outer = [], []
    for j in range(ny):
        for i in range(nx):
            a = j*nx+i
            box = cells[a]["bounds"]
            for b, axis, fa, fb in [(a+1, 0, 1, 0), (a+nx, 1, 3, 2)]:
                if (axis == 0 and i == nx-1) or (axis == 1 and j == ny-1):
                    continue
                lo, hi = box[2:] if axis == 0 else box[:2]
                endpoints = ([[box[fa], lo], [box[fa], hi]] if axis == 0 else
                             [[lo, box[fa]], [hi, box[fa]]])
                edges.append({"a": a, "b": b, "axis": axis, "fa": fa, "fb": fb,
                              "interface": cells[a]["material"] != cells[b]["material"],
                              "normal": [1, 0] if axis == 0 else [0, 1],
                              "tangent": [0, 1] if axis == 0 else [1, 0],
                              "endpoints": endpoints, "length": hi-lo})
            for face, side, normal, active in [
                (0, "left", [-1, 0], i == 0), (1, "right", [1, 0], i == nx-1),
                (2, "bottom", [0, -1], j == 0), (3, "top", [0, 1], j == ny-1)
            ]:
                if active:
                    outer.append({"cell": a, "face": face, "side": side, "normal": normal,
                                  "length": box[3]-box[2] if face < 2 else box[1]-box[0],
                                  "uy": .5 if side == "bottom" else -.5 if side == "top" else None})
    corners = []
    # Quadrants NE, NW, SW, SE, in counterclockwise order. Adjacent midpoints
    # establish each sector from actual cells, including both reentrant phases.
    for j in range(1, len(ys)-1):
        for i in range(1, len(xs)-1):
            quadrants = [j*nx+i, j*nx+i-1, (j-1)*nx+i-1, (j-1)*nx+i]
            materials = [cells[k]["material"] for k in quadrants]
            count = materials.count("grout")
            if count not in (1, 3):
                continue
            reentrant = "grout" if count == 3 else "soil"
            missing = next(k for k, m in enumerate(materials) if m != reentrant)
            corners.append({"id": f"corner_{len(corners)}", "point": [xs[i], ys[j]],
                            "quadrant_materials": materials, "reentrant_material": reentrant,
                            "rotation_quadrants": missing, "opening_angle": 1.5*np.pi,
                            "role": "junction" if count == 3 else "tip"})
    for corner in corners:
        separation = min(np.linalg.norm(np.subtract(corner["point"], other["point"]))
                         for other in corners if other is not corner)
        corner["mode_radius"] = .9*min(spec["arm_width"], separation)
    area = sum((c["bounds"][1]-c["bounds"][0])*(c["bounds"][3]-c["bounds"][2])
               for c in cells if c["material"] == "grout")
    return {**spec, "cells": cells, "edges": edges, "corners": corners,
            "outer_boundaries": outer, "grout_area": area,
            "horizontal_datum": {"point": [x0, y0], "cell": 0, "ux": 0.}}


def observation_supports(geometry):
    """Choose junction, branch and tip supports before inspecting a solution."""
    segments = []
    length = geometry["arm_width"]/2
    for ie, edge in enumerate(geometry["edges"]):
        if not edge["interface"]:
            continue
        p0, p1 = np.array(edge["endpoints"])
        segments.append({"id": f"edge_{ie}", "edge": ie, "role": "whole_interface",
                         "endpoints": [p0.tolist(), p1.tolist()]})
        for corner in geometry["corners"]:
            center = np.array(corner["point"])
            for start, end in [(p0, p1), (p1, p0)]:
                if np.allclose(start, center, atol=1e-14, rtol=0.):
                    stop = start+(end-start)*min(length/edge["length"], .5)
                    segments.append({"id": f"{corner['id']}_edge_{ie}", "edge": ie,
                                     "corner_id": corner["id"], "role": corner["role"],
                                     "endpoints": [start.tolist(), stop.tolist()]})
    # All boxes are tied to the actual cap/stem endpoints and clipped later to soil.
    cap, stem = geometry["rectangles"]
    w = geometry["arm_width"]
    regions = [
        {"id": "above_cap", "bounds": [stem[0]-w/2, stem[1]+w/2, cap[3], cap[3]+w]},
        {"id": "below_stem", "bounds": [stem[0]-w/2, stem[1]+w/2, stem[2]-w, stem[2]]},
        {"id": "left_shoulder", "bounds": [stem[0]-w, stem[0], cap[2]-w, cap[2]]},
        {"id": "right_shoulder", "bounds": [stem[1], stem[1]+w, cap[2]-w, cap[2]]},
    ]
    return {"segments": segments, "soil_regions": regions,
            "section_y": [stem[2]-w/2, stem[2]+w/2, cap[2]-w/2,
                          (cap[2]+cap[3])/2, cap[3]+w/2],
            "selection_reason": "Half-width corner segments, cap/stem tip neighborhoods, adjacent soil shoulders and sections along the axial load path."}
