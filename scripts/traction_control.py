"""Consistent equilibrium and traction control for the displacement PINN.

Add squared governing-equation residuals to the original potential energy.
Constitutive stress remains C:epsilon(u); no separate stress fit is introduced.
The characteristic length is the physical arm width. The interface metric
can use material compliance L/E or the spring's complementary energy K^-1.
Equilibrium can use the same reference modulus in both materials so that a
given force residual is measured equally on each side of the junction.
"""
from __future__ import annotations

import numpy as np
import torch

from full_domain_pinn import full_domain_quadrature, interface_readouts
from full_domain_geometry import observation_supports
from local_virtual_work import strain_of
from partitioned_mixed_pinn import gauss, sample_one


def stress_divergence(stress, xy):
    """Evaluate spatial derivatives for fixed-feature coefficient assembly."""
    # The zero term keeps the derivative defined for constant stress fields.
    derivatives = [torch.autograd.grad((stress[..., j]+0*xy[..., 0]).sum(), xy,
                                      retain_graph=j < 2)[0] for j in range(3)]
    return torch.stack([derivatives[0][..., 0]+derivatives[2][..., 1],
                        derivatives[2][..., 0]+derivatives[1][..., 1]], -1)


class EquilibriumControl:
    def __init__(self, model, nodes, edge_nodes, support_nodes=None, length=None,
                 interface_metric="material", equilibrium_metric="material"):
        self.model = model
        self.length = model.vein_width if length is None else length
        if self.length <= 0:
            raise ValueError("The equilibrium control length must be positive.")
        if interface_metric not in ("material", "spring"):
            raise ValueError(f"Unknown interface metric: {interface_metric}")
        self.interface_metric = interface_metric
        if equilibrium_metric not in ("material", "reference"):
            raise ValueError(f"Unknown equilibrium metric: {equilibrium_metric}")
        self.equilibrium_metric = equilibrium_metric
        self.quad = list(full_domain_quadrature(model, nodes, edge_nodes, support_nodes))
        self.quad[0].requires_grad_(True)
        self.quad[2].requires_grad_(True)

    def bulk_blocks(self, displacement):
        """Evaluate interior equilibrium without evaluating any face residual."""
        model, length = self.model, self.length
        xy, weights = self.quad[:2]
        sigma = model.stress(strain_of(displacement(xy), xy))
        divergence = stress_divergence(sigma, xy)
        # Stresses and moduli are nondimensionalized by E_s. The common
        # reference modulus is therefore one, while the constitutive law
        # above continues to use each material's actual stiffness.
        equation_moduli = (model.E[:, 0, 0] if self.equilibrium_metric == "material"
                          else torch.ones_like(model.E[:, 0, 0]))
        bulk = (weights/equation_moduli[:, None]).sqrt()[..., None]*length*divergence
        return [("bulk_equilibrium", cell["material"], bulk[i][weights[i] > 0].flatten())
                for i, cell in enumerate(model.cells)]

    def bulk_residual(self, displacement):
        return torch.cat([value for _, _, value in self.bulk_blocks(displacement)])

    def blocks(self, displacement):
        """Return equation/material blocks without losing either interface side."""
        model, length = self.model, self.length
        faces, wf = self.quad[2:]
        equation_moduli = (model.E[:, 0, 0] if self.equilibrium_metric == "material"
                          else torch.ones_like(model.E[:, 0, 0]))
        blocks = self.bulk_blocks(displacement)
        uv = displacement(faces)
        stress = model.stress(strain_of(uv, faces)).reshape(len(model.cells), 4, -1, 3)
        uv = uv.reshape(len(model.cells), 4, -1, 2)
        young = model.E[:, 0, 0]
        for edge in model.edges:
            a, b, axis, fa, fb = [edge[k] for k in ("a", "b", "axis", "fa", "fb")]
            components = [0, 2] if axis == 0 else [2, 1]
            ta, tb = stress[a, fa][:, components], stress[b, fb][:, components]
            if edge["interface"]:
                k = model.k if axis == 0 else model.k.flip(0)
                spring = k*(uv[b, fb]-uv[a, fa])
                for cell, traction in ((a, ta), (b, tb)):
                    compliance = 1/k if self.interface_metric == "spring" else length/young[cell]
                    residual = (wf[a, fa, :, None]*compliance).sqrt()*(traction-spring)
                    blocks.append(("interface_traction", model.cells[cell]["material"], residual.flatten()))
            else:
                residual = (length*wf[a, fa]/(equation_moduli[a]+equation_moduli[b])).sqrt()[:, None]*(ta-tb)
                blocks.append(("shared_traction", model.cells[a]["material"], residual.flatten()))
        for edge in model.geometry["outer_boundaries"]:
            cell, face = edge["cell"], edge["face"]
            # Lateral faces are free; platens prescribe only u_y.
            components = [0, 2] if face < 2 else [2]
            residual = (length*wf[cell, face]/equation_moduli[cell]).sqrt()[:, None]*stress[cell, face][:, components]
            blocks.append(("natural_traction", model.cells[cell]["material"], residual.flatten()))
        return blocks

    def terms(self, displacement):
        blocks = self.blocks(displacement)
        return {name: torch.cat([value for key, _, value in blocks if key == name])
                for name in ("bulk_equilibrium", "interface_traction", "shared_traction", "natural_traction")}

    def __call__(self, displacement):
        return torch.cat(list(self.terms(displacement).values()))

    def readout(self):
        names = ("bulk_equilibrium", "interface_traction", "shared_traction", "natural_traction")
        materials = {name: dict.fromkeys(names, 0.) for name in ("soil", "grout")}
        for name, material, value in self.blocks(self.model.displacement):
            materials[material][name] += float(.5*value.detach().square().sum())
        values = {name: sum(row[name] for row in materials.values()) for name in names}
        return {"length_over_H": self.length, "interface_metric": self.interface_metric,
                "equilibrium_metric": self.equilibrium_metric,
                "penalty_terms": values, "material_penalty_terms": materials,
                "total_penalty": sum(values.values()),
                "bulk_points_per_cell": (self.quad[1] > 0).sum(1).tolist()}


def junction_balance_readout(model, n=128, radius=None):
    """Balance a grout L-shaped patch around each junction corner.

    Keep exterior constitutive and spring-substituted balances separately.
    Internal same-material faces are paired and reported as a traction jump.
    All returned forces use N per mm out-of-plane thickness.
    """
    z, w = gauss(n)
    radius = model.vein_width/2 if radius is None else radius
    scale = model.case.soil_E_MPa*model.case.applied_shortening_mm
    edge_by_face = {}
    for edge in model.edges:
        edge_by_face[(edge["a"], edge["fa"])] = edge
        edge_by_face[(edge["b"], edge["fb"])] = edge
    rows = []
    for corner in model.geometry["corners"]:
        if corner["role"] != "junction":
            continue
        cx, cy = corner["point"]
        patches = {}
        for i, cell in enumerate(model.cells):
            if cell["material"] != "grout":
                continue
            a, b, c, d = cell["bounds"]
            bounds = [max(a, cx-radius), min(b, cx+radius), max(c, cy-radius), min(d, cy+radius)]
            if bounds[1]-bounds[0] > 1e-12 and bounds[3]-bounds[2] > 1e-12:
                patches[i] = bounds
        external, substituted, internal = np.zeros(2), np.zeros(2), np.zeros(2)
        face_rows = []
        for i, (a, b, c, d) in patches.items():
            for face in range(4):
                axis, sign = face//2, (-1 if face % 2 == 0 else 1)
                position = (a, b, c, d)[face]
                lo, hi = (c, d) if axis == 0 else (a, b)
                points = np.empty((n, 2))
                points[:, axis] = position
                points[:, 1-axis] = lo+(hi-lo)*z
                p = sample_one(model, i, points)
                components = [2, 4] if axis == 0 else [4, 3]
                force = sign*(hi-lo)*(w[:, None]*p[:, components]).sum(0)*scale
                cell_face = abs(position-model.cells[i]["bounds"][face]) < 1e-12
                edge = edge_by_face.get((i, face)) if cell_face else None
                inside = edge is not None and not edge["interface"] and edge["a"] in patches and edge["b"] in patches
                if inside:
                    internal += force
                    role = "internal_shared_face"
                    spring_force = None
                else:
                    external += force
                    if edge is not None and edge["interface"]:
                        pa, pb = sample_one(model, edge["a"], points), sample_one(model, edge["b"], points)
                        k = model.k.detach().numpy() if axis == 0 else model.k.flip(0).detach().numpy()
                        spring_force = sign*(hi-lo)*(w[:, None]*k*(pb[:, :2]-pa[:, :2])).sum(0)*scale
                        substituted += spring_force
                        role = "physical_interface"
                    else:
                        spring_force = None
                        substituted += force
                        role = "interior_section"
                endpoints = [[position, lo], [position, hi]] if axis == 0 else [[lo, position], [hi, position]]
                face_rows.append({"cell": i, "face": face, "role": role,
                                  "endpoints": endpoints,
                                  "constitutive_force_xy": force.tolist(),
                                  "spring_force_xy": None if spring_force is None else spring_force.tolist()})
        rows.append({"corner_id": corner["id"], "center_over_H": [cx, cy],
                     "radius_over_H": radius, "grout_area_over_H2": sum((b-a)*(d-c) for a, b, c, d in patches.values()),
                     "force_units": "N/mm thickness", "constitutive_boundary_net_xy": external.tolist(),
                     "spring_substituted_boundary_net_xy": substituted.tolist(),
                     "internal_cut_net_xy": internal.tolist(), "faces": face_rows})
    return rows


def junction_scale_readout(model, radii_over_arm_width, n=256):
    """Read neighboring physical scales without adding them to the fit."""
    original = [s for s in observation_supports(model.geometry)["segments"] if s["role"] == "junction"]
    rows = []
    for relative_radius in radii_over_arm_width:
        radius = relative_radius*model.vein_width
        segments = []
        for support in original:
            start, stop = np.asarray(support["endpoints"])
            direction = (stop-start)/np.linalg.norm(stop-start)
            if radius > model.edges[support["edge"]]["length"]:
                raise ValueError("The diagnostic segment exceeds its physical interface.")
            segments.append({**support, "endpoints": [start.tolist(), (start+radius*direction).tolist()]})
        rows.append({"radius_over_arm_width": relative_radius,
                     "segments": interface_readouts(model, segments, n),
                     "regions": [{key: value for key, value in region.items() if key != "faces"}
                                 for region in junction_balance_readout(model, n, radius)]})
    return rows
