"""Compatible neural Ritz field on a complete, non-mirrored grout/soil domain.

Reuses the energy operator and coefficient solve of partitioned_energy_pinn.
No reflection or symmetry-plane constraint enters this displacement space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import brentq
import torch
from torch import nn

from full_domain_geometry import OFFSET_T, build_geometry, observation_supports
from junction_enrichment import wedge_displacement
from local_virtual_work import displacement_residual
from partitioned_energy_pinn import EnergyNet, energy_residual
from partitioned_mixed_pinn import (SharedTrace, gauss, quadrature,
                                    region_mean_strain, sample_one, traction)


@dataclass
class FullDomainCase:
    geometry: str = "offset_t"
    soil_E_MPa: float = 79.9
    grout_E_MPa: float = 12700.
    poisson: float = .3
    specimen_size_mm: float = 100.
    applied_shortening_mm: float = .1
    normal_K_MPa_per_mm: float = 10.
    tangent_K_MPa_per_mm: float = 10.


def platen_factor(y, bounds):
    lo, hi = bounds[2:]
    return 4*(y-lo)*(hi-y)/(hi-lo)**2


class AdmissibleCorrection(nn.Module):
    """Project a material-shared correction onto the homogeneous platen/gauge BCs."""
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.register_buffer("material_ids", torch.tensor(
            [int(c["material"] == "grout") for c in geometry["cells"]]))
        self.register_buffer("datum", torch.tensor(geometry["horizontal_datum"]["point"]))

    def forward(self, xy):
        result = self.raw(xy)
        datum_xy = self.datum.reshape(1, 1, 2).expand(len(self.material_ids), 1, 2)
        anchor = self.raw(datum_xy)[self.geometry["horizontal_datum"]["cell"], 0, 0]
        # A single global rigid translation changes neither strain nor jumps.
        return torch.stack([result[..., 0]-anchor,
                            result[..., 1]*platen_factor(xy[..., 1], self.geometry["bounds"])], -1)


class OrientedCornerModes(AdmissibleCorrection):
    """Independent 270-degree leading modes in each actual reentrant material.

Finite positive springs introduce a length E/K. As r tends to zero, a
singular stress r**(lambda-1), 0<lambda<1, must have traction-free leading
faces; a finite spring jump has no matching singular order. The *complete*
field still obeys the finite-spring variational law, not a free-face model.
"""
    def __init__(self, case, geometry):
        super().__init__(geometry)
        self.nu = case.poisson
        self.exponents = (brentq(lambda z: np.sin(1.5*np.pi*z)-z, .5, .6),
                          brentq(lambda z: np.sin(1.5*np.pi*z)+z, .8, .95))
        self.coefficients = nn.Parameter(torch.zeros(len(geometry["corners"]), 2))
        self.register_buffer("scale", torch.tensor([1., np.sqrt(case.soil_E_MPa/case.grout_E_MPa)]))

    def raw(self, xy):
        result = 0*xy.square()
        for ic, corner in enumerate(self.geometry["corners"]):
            # Fixed-feature assembly activates one corner at a time. Retain
            # zero-coefficient modes when their coefficients are trainable.
            if not self.coefficients.requires_grad and not self.coefficients[ic].any():
                continue
            angle = corner["rotation_quadrants"]*np.pi/2
            rotation = xy.new_tensor([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]]).round()
            local = (xy-xy.new_tensor(corner["point"]))@rotation
            r2 = local.square().sum(-1)
            radius = corner["mode_radius"]
            material = int(corner["reentrant_material"] == "grout")
            active = ((self.material_ids[:, None] == material) & (r2 > 0.) & (r2 < radius**2))
            # Inactive cells must not evaluate a singular vertex or angular cut.
            x = torch.where(active, local[..., 0], torch.ones_like(r2))
            y = torch.where(active, local[..., 1], -torch.ones_like(r2))
            modes = torch.stack([wedge_displacement(x, y, lam, j == 0, self.nu)
                                 for j, lam in enumerate(self.exponents)], -2)@rotation.T
            window = torch.relu(1-r2/radius**2)**3*active
            value = (modes*self.coefficients[ic, :, None]).sum(-2)
            # Normalize the platen multiplier to retain the local leading mode.
            factor = platen_factor(corner["point"][1], self.geometry["bounds"])
            value = value*xy.new_tensor([1., 1/factor])
            result = result+window[..., None]*value*self.scale[material]
        return result


class IndependentLocalModes(AdmissibleCorrection):
    """Shared smooth traces with optional independent, trace-free cell interiors."""
    def __init__(self, case, geometry, radius=.1, width=6, seed=17, split_interior=False,
                 cellwise_interior=False):
        super().__init__(geometry)
        if radius <= 0.:
            raise ValueError("The local support radius must be positive.")
        if cellwise_interior and not split_interior:
            raise ValueError("Cellwise interiors require split interior readouts.")
        self.radius, self.width, self.seed = radius, width, seed
        self.cellwise_interior = cellwise_interior
        self.centers = [c["point"] for c in geometry["corners"] if c["role"] == "junction"]
        generator = torch.Generator().manual_seed(seed)
        count = len(self.centers)
        self.w1 = nn.Parameter(1.5*torch.randn(count, 2, 2, width, generator=generator))
        self.b1 = nn.Parameter(.5*torch.randn(count, 2, 1, width, generator=generator))
        self.coefficients = nn.Parameter(torch.zeros(count, 2, width+1, 2))
        interior_count = len(geometry["cells"]) if cellwise_interior else 2
        self.interior_coefficients = (nn.Parameter(torch.zeros(count, interior_count, width+1, 2))
                                      if split_interior else None)
        self.register_buffer("scale", torch.tensor([1., np.sqrt(case.soil_E_MPa/case.grout_E_MPa)]))

    def raw(self, xy):
        result = torch.zeros_like(xy)
        ids = self.material_ids
        for ic, center in enumerate(self.centers):
            local = (xy-xy.new_tensor(center))/self.radius
            window = torch.relu(1-local.square().sum(-1, keepdim=True))**3
            hidden = torch.tanh(torch.bmm(local, self.w1[ic, ids])+self.b1[ic, ids])
            features = torch.cat([torch.ones_like(hidden[..., :1]), hidden], -1)
            value = torch.bmm(features, self.coefficients[ic, ids])
            if self.interior_coefficients is not None:
                bounds = xy.new_tensor([cell["bounds"] for cell in self.geometry["cells"]])[:, None]
                a = (xy[..., 0:1]-bounds[..., 0:1])/(bounds[..., 1:2]-bounds[..., 0:1])
                b = (xy[..., 1:2]-bounds[..., 2:3])/(bounds[..., 3:4]-bounds[..., 2:3])
                bubble = 16*a*(1-a)*b*(1-b)
                readout = (self.interior_coefficients[ic] if self.cellwise_interior
                           else self.interior_coefficients[ic, ids])
                value = value+bubble*torch.bmm(features, readout)
            factor = platen_factor(center[1], self.geometry["bounds"])
            result = result+window*value*xy.new_tensor([1., 1/factor])
        return result*self.scale[ids, None, None]

    def interior_readout(self, xy):
        """Evaluate the split interior basis on its active corner/cell pairs."""
        result = 0*xy.square()
        bounds = xy.new_tensor([cell["bounds"] for cell in self.geometry["cells"]])[:, None]
        for ic, center in enumerate(self.centers):
            coefficients = (self.interior_coefficients[ic] if self.cellwise_interior
                            else self.interior_coefficients[ic, self.material_ids])
            cells = torch.where(coefficients.detach().flatten(1).any(-1))[0]
            if not len(cells):
                continue
            points, box, ids = xy[cells], bounds[cells], self.material_ids[cells]
            local = (points-xy.new_tensor(center))/self.radius
            hidden = torch.tanh(torch.bmm(local, self.w1[ic, ids])+self.b1[ic, ids])
            features = torch.cat([torch.ones_like(hidden[..., :1]), hidden], -1)
            a = (points[..., :1]-box[..., :1])/(box[..., 1:2]-box[..., :1])
            b = (points[..., 1:2]-box[..., 2:3])/(box[..., 3:4]-box[..., 2:3])
            window = torch.relu(1-local.square().sum(-1, keepdim=True))**3
            value = 16*a*(1-a)*b*(1-b)*window*torch.bmm(features, coefficients[cells])
            vertical = (platen_factor(points[..., 1], self.geometry["bounds"])
                        /platen_factor(center[1], self.geometry["bounds"]))
            value = value*torch.stack([torch.ones_like(vertical), vertical], -1)
            value = value*self.scale[ids, None, None]
            result = result.index_add(0, cells, value)
        # The cell bubble is zero at the corner datum, so its horizontal
        # anchor subtraction in AdmissibleCorrection is identically zero.
        return result


def junction_shared_supports(geometry):
    """Same-material cuts meeting an actual grout junction."""
    supports = []
    for corner in geometry["corners"]:
        if corner["role"] != "junction":
            continue
        for index, edge in enumerate(geometry["edges"]):
            if not edge["interface"] and corner["point"] in edge["endpoints"]:
                endpoint = edge["endpoints"].index(corner["point"])
                supports.append({"id": f"{corner['id']}_shared_edge_{index}", "edge": index,
                                 "corner_id": corner["id"],
                                 "endpoints": [corner["point"], edge["endpoints"][1-endpoint]]})
    return supports


class InterfaceLocalModes(nn.Module):
    """Compact face fields with independent or same-material shared traces."""
    def __init__(self, case, geometry, radius=.05, width=6, seed=43, shared=False):
        super().__init__()
        if radius <= 0.:
            raise ValueError("The interface support radius must be positive.")
        self.radius, self.shared = radius, shared
        supports = (junction_shared_supports(geometry) if shared else
                    [s for s in observation_supports(geometry)["segments"] if s["role"] == "junction"])
        self.sides = [{"support": support, "side": side,
                       "cell": geometry["edges"][support["edge"]][side],
                       "face": geometry["edges"][support["edge"]]["f"+side]}
                      for support in supports for side in ("a", "b")]
        count = len(supports) if shared else len(self.sides)
        if shared:
            self.register_buffer("support_ids", torch.arange(count).repeat_interleave(2))
        self.register_buffer("cell_ids", torch.tensor([s["cell"] for s in self.sides]))
        self.register_buffer("faces", torch.tensor([s["face"] for s in self.sides]))
        self.register_buffer("bounds", torch.tensor([geometry["cells"][s["cell"]]["bounds"] for s in self.sides])[:, None])
        self.register_buffer("centers", torch.tensor([s["support"]["endpoints"][0] for s in self.sides])[:, None])
        self.register_buffer("scale", torch.tensor([
            np.sqrt(case.soil_E_MPa/case.grout_E_MPa)
            if geometry["cells"][s["cell"]]["material"] == "grout" else 1.
            for s in self.sides])[:, None, None])
        generator = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(1.5*torch.randn(count, 1, width, generator=generator))
        self.b1 = nn.Parameter(.5*torch.randn(count, 1, width, generator=generator))
        self.coefficients = nn.Parameter(torch.zeros(count, width+2, 2))

    def forward(self, xy):
        points = xy[self.cell_ids]
        a = (points[..., 0:1]-self.bounds[..., 0:1])/(self.bounds[..., 1:2]-self.bounds[..., 0:1])
        b = (points[..., 1:2]-self.bounds[..., 2:3])/(self.bounds[..., 3:4]-self.bounds[..., 2:3])
        vertical = (self.faces < 2)[:, None, None]
        z = torch.where(vertical, b, a)
        normal_coordinate = torch.where(vertical, a, b)
        extension = torch.where((self.faces % 2 == 0)[:, None, None], 1-normal_coordinate, normal_coordinate)
        local = (points-self.centers)/self.radius
        tangent = torch.where(vertical, local[..., 1:2], local[..., 0:1])
        w1 = self.w1[self.support_ids] if self.shared else self.w1
        b1 = self.b1[self.support_ids] if self.shared else self.b1
        readout = self.coefficients[self.support_ids] if self.shared else self.coefficients
        hidden = torch.tanh(torch.bmm(tangent, w1)+b1)
        features = torch.cat([torch.ones_like(tangent), tangent, hidden], -1)
        window = torch.relu(1-local.square().sum(-1, keepdim=True))**3
        value = 4*z*(1-z)*extension*window*torch.bmm(features, readout)*self.scale
        return torch.zeros_like(xy).index_add(0, self.cell_ids, value)


class CellInteriorModes(nn.Module):
    """Compact zero-trace fields with independent junction/cell readouts."""
    def __init__(self, case, geometry, radius=.05, width=6, seed=47):
        super().__init__()
        if radius <= 0.:
            raise ValueError("The interior support radius must be positive.")
        self.radius, self.geometry = radius, geometry
        centers = [c["point"] for c in geometry["corners"] if c["role"] == "junction"]
        self.supports = []
        for ic, center in enumerate(centers):
            for cell, entry in enumerate(geometry["cells"]):
                x0, x1, y0, y1 = entry["bounds"]
                distance2 = (max(x0-center[0], 0., center[0]-x1)**2
                             +max(y0-center[1], 0., center[1]-y1)**2)
                if distance2 < radius**2:
                    self.supports.append({"center": center, "corner": ic, "cell": cell})
        self.register_buffer("cell_ids", torch.tensor([s["cell"] for s in self.supports]))
        self.register_buffer("corner_ids", torch.tensor([s["corner"] for s in self.supports]))
        self.register_buffer("material_ids", torch.tensor([
            int(geometry["cells"][s["cell"]]["material"] == "grout") for s in self.supports]))
        self.register_buffer("bounds", torch.tensor([
            geometry["cells"][s["cell"]]["bounds"] for s in self.supports])[:, None])
        self.register_buffer("centers", torch.tensor([s["center"] for s in self.supports])[:, None])
        self.register_buffer("scale", torch.tensor([1., np.sqrt(case.soil_E_MPa/case.grout_E_MPa)]))
        generator = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(1.5*torch.randn(len(centers), 2, 2, width, generator=generator))
        self.b1 = nn.Parameter(.5*torch.randn(len(centers), 2, 1, width, generator=generator))
        self.coefficients = nn.Parameter(torch.zeros(len(self.supports), width+1, 2))

    def forward(self, xy):
        points = xy[self.cell_ids]
        a = (points[..., 0:1]-self.bounds[..., 0:1])/(self.bounds[..., 1:2]-self.bounds[..., 0:1])
        b = (points[..., 1:2]-self.bounds[..., 2:3])/(self.bounds[..., 3:4]-self.bounds[..., 2:3])
        local = (points-self.centers)/self.radius
        hidden = torch.tanh(torch.bmm(local, self.w1[self.corner_ids, self.material_ids])
                            +self.b1[self.corner_ids, self.material_ids])
        features = torch.cat([torch.ones_like(hidden[..., :1]), hidden], -1)
        window = torch.relu(1-local.square().sum(-1, keepdim=True))**3
        value = 16*a*(1-a)*b*(1-b)*window*torch.bmm(features, self.coefficients)
        vertical = (platen_factor(points[..., 1], self.geometry["bounds"])
                    /platen_factor(self.centers[..., 1], self.geometry["bounds"]))
        value = value*torch.stack([torch.ones_like(vertical), vertical], -1)
        value = value*self.scale[self.material_ids, None, None]
        return torch.zeros_like(xy).index_add(0, self.cell_ids, value)


class FullDomainNet(EnergyNet):
    """Coons trace field, cell interiors and independent local corrections."""
    def __init__(self, case=None, geometry_spec=OFFSET_T, width=12, edge_width=8,
                 seed=7, corners=True, local=None, additional_local=None, interface_local=None,
                 cell_interior=None, shared_local=None, junction_patches=None):
        nn.Module.__init__(self)
        self.case = case or FullDomainCase()
        if min(self.case.normal_K_MPa_per_mm, self.case.tangent_K_MPa_per_mm) <= 0.:
            raise ValueError("Both interface stiffnesses must be finite and positive.")
        if not np.isfinite([self.case.normal_K_MPa_per_mm, self.case.tangent_K_MPa_per_mm]).all():
            raise ValueError("Both interface stiffnesses must be finite and positive.")
        self.geometry = build_geometry(geometry_spec)
        self.cells, self.edges = self.geometry["cells"], self.geometry["edges"]
        self.vein_width = self.geometry["arm_width"]
        self.full_domain = True
        self.corner_enrichment = self.elastic_corners = False
        self.options = {"geometry_spec": geometry_spec, "width": width, "edge_width": edge_width,
                        "seed": seed, "corners": corners, "local": local,
                        "additional_local": additional_local, "interface_local": interface_local,
                        "cell_interior": cell_interior, "shared_local": shared_local,
                        "junction_patches": junction_patches}
        torch.manual_seed(seed)
        nc = len(self.cells)
        bounds = torch.tensor([c["bounds"] for c in self.cells])
        self.register_buffer("bounds", bounds[:, None, :])
        self.register_buffer("E", torch.tensor([
            self.case.grout_E_MPa/self.case.soil_E_MPa if c["material"] == "grout" else 1.
            for c in self.cells])[:, None, None])
        self.register_buffer("k", torch.tensor([self.case.normal_K_MPa_per_mm,
                                                self.case.tangent_K_MPa_per_mm])
                             *self.case.specimen_size_mm/self.case.soil_E_MPa)
        self.register_buffer("displacement_scale", torch.minimum(
            bounds[:, 1]-bounds[:, 0], bounds[:, 3]-bounds[:, 2])[:, None, None]/self.E.sqrt())
        self.weights, self.biases = nn.ParameterList(), nn.ParameterList()
        for n0, n1 in [(2, width), (width, width), (width, 2)]:
            weight = torch.empty(nc, n0, n1)
            for patch in weight:
                nn.init.xavier_normal_(patch)
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(nc, 1, n1)))
        nn.init.zeros_(self.weights[-1])
        self.trace = SharedTrace(self.cells, self.case, "compliance", True, edge_width)
        self.junction = OrientedCornerModes(self.case, self.geometry) if corners else None
        self.local = IndependentLocalModes(self.case, self.geometry, **local) if local else None
        self.additional_local = (IndependentLocalModes(self.case, self.geometry, **additional_local)
                                 if additional_local else None)
        self.interface_local = (InterfaceLocalModes(self.case, self.geometry, **interface_local)
                                if interface_local else None)
        self.cell_interior = (CellInteriorModes(self.case, self.geometry, **cell_interior)
                              if cell_interior else None)
        self.shared_local = (InterfaceLocalModes(self.case, self.geometry, shared=True, **shared_local)
                             if shared_local else None)
        self.junction_patches = nn.ModuleList([
            self.make_junction_patch(settings) for settings in (junction_patches or [])])

    def make_junction_patch(self, settings):
        return nn.ModuleDict({
            "interface": InterfaceLocalModes(self.case, self.geometry, **settings),
            "shared": InterfaceLocalModes(self.case, self.geometry, shared=True, **settings),
            "interior": CellInteriorModes(self.case, self.geometry, **settings)})

    def compact_components(self):
        for component in (self.interface_local, self.cell_interior, self.shared_local):
            if component is not None:
                yield component
        for patch in self.junction_patches:
            yield from patch.values()

    def displacement(self, xy, homogeneous=False, include_enrichment=True):
        b = self.bounds
        a = (xy[..., 0:1]-b[..., 0:1])/(b[..., 1:2]-b[..., 0:1])
        c = (xy[..., 1:2]-b[..., 2:3])/(b[..., 3:4]-b[..., 2:3])
        hidden = 2*torch.cat([a, c], -1)-1
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = torch.bmm(hidden, weight)+bias
            if i < 2:
                hidden = torch.tanh(hidden)
        uv = self.trace(a, c, hidden*self.displacement_scale, homogeneous)
        if include_enrichment:
            if self.junction is not None:
                uv = uv+self.junction(xy)
            if self.local is not None:
                uv = uv+self.local(xy)
            if self.additional_local is not None:
                uv = uv+self.additional_local(xy)
            for component in self.compact_components():
                uv = uv+component(xy)
        return uv

    def save(self, path):
        torch.save({"case": asdict(self.case), "options": self.options,
                    "state_dict": self.state_dict()}, path)

    def enrichment_parameters(self):
        linear = super().enrichment_parameters()
        for local in (self.local, self.additional_local):
            if local is not None and local.interior_coefficients is not None:
                linear.append(local.interior_coefficients)
        for component in self.compact_components():
            linear.append(component.coefficients)
        return linear

    def readout_displacement(self, parameter):
        if any(parameter is p for p in (self.weights[-1], self.biases[-1],
                                        self.trace.node_values, self.trace.w2, self.trace.b2)):
            return lambda xy: self.base_readout_displacement(parameter, xy)
        for component in (self.local, self.additional_local):
            if component is not None and parameter is component.interior_coefficients:
                return component.interior_readout
        for component in self.compact_components():
            if parameter is component.coefficients:
                return component
        return super().readout_displacement(parameter)

    def base_readout_displacement(self, parameter, xy):
        """Evaluate only cells touched by the active base readout coefficients.

        This is the same homogeneous basis used by displacement(), with the
        other readouts zero. It avoids differentiating zero fields in unrelated
        cells during the coefficient-by-coefficient least-squares assembly.
        """
        trace = self.trace
        interior = parameter is self.weights[-1] or parameter is self.biases[-1]
        nodes = parameter is trace.node_values
        active = parameter.detach().reshape(parameter.shape[0], -1).any(-1)
        if not interior:
            active = active[trace.corners if nodes else trace.edge_ids].any(-1)
        cells = torch.where(active)[0]
        points, bounds = xy[cells], self.bounds[cells]
        a = (points[..., :1]-bounds[..., :1])/(bounds[..., 1:2]-bounds[..., :1])
        b = (points[..., 1:2]-bounds[..., 2:3])/(bounds[..., 3:4]-bounds[..., 2:3])
        if interior:
            if parameter is self.weights[-1]:
                hidden = 2*torch.cat([a, b], -1)-1
                for weight, bias in zip(self.weights[:2], self.biases[:2]):
                    hidden = torch.tanh(torch.bmm(hidden, weight[cells])+bias[cells])
                value = torch.bmm(hidden, parameter[cells])
            else:
                value = parameter[cells].expand(-1, points.shape[1], -1)
            value = 16*a*(1-a)*b*(1-b)*value*self.displacement_scale[cells]
        elif nodes:
            values = (parameter*trace.node_scale*trace.node_mask)[trace.corners[cells]]
            value = ((1-a)*(1-b)*values[:, None, 0]+a*(1-b)*values[:, None, 1]
                     +(1-a)*b*values[:, None, 2]+a*b*values[:, None, 3])
        else:
            value = torch.zeros_like(points)
            for face, z, extension in ((0, b, 1-a), (1, b, a), (2, a, 1-b), (3, a, b)):
                ids = trace.edge_ids[cells, face]
                if parameter is trace.w2:
                    hidden = torch.tanh(torch.bmm(2*z-1, trace.w1[ids])+trace.b1[ids])
                    edge = torch.bmm(hidden, parameter[ids])
                else:
                    edge = parameter[ids].expand(-1, points.shape[1], -1)
                value = value+extension*z*(1-z)*edge*trace.edge_mask[ids]*trace.edge_scale[ids]
        return (0*xy.square()).index_add(0, cells, value)


def load_full_model(path):
    saved = torch.load(path, weights_only=False)
    model = FullDomainNet(FullDomainCase(**saved["case"]), **saved["options"])
    model.load_state_dict(saved["state_dict"])
    return model


def full_domain_quadrature(model, nodes, edge_nodes, support_nodes=None, extra_supports=()):
    """Resolve compact face/interior fields at the existing bulk density.

    Split affected cell axes at the projections of each compact support.
    Each interval receives its share of the original bulk order, with at
    least support_nodes points. Pad shorter cell rules with zero weights so
    the existing batched displacement and energy operators remain usable.
    Face integration is unchanged. extra_supports supplies (cell, center, radius)
    for diagnostic variations that are not present in the saved trial field.
    """
    if support_nodes is None:
        return quadrature(model.cells, nodes, edge_nodes)
    components = list(model.compact_components())
    if (not extra_supports and not components) or support_nodes < 2:
        raise ValueError("Support quadrature requires compact local fields and at least two nodes per interval.")
    supports = list(extra_supports)
    for component in components:
        if isinstance(component, InterfaceLocalModes):
            supports.extend((s["cell"], s["support"]["endpoints"][0], component.radius)
                            for s in component.sides)
        else:
            supports.extend((s["cell"], s["center"], component.radius) for s in component.supports)
    points, weights = [], []
    for index, cell in enumerate(model.cells):
        centers = [(center, radius) for cell_id, center, radius in supports if cell_id == index]
        axes = []
        for axis in (0, 1):
            lo, hi = cell["bounds"][2*axis:2*axis+2]
            candidates = [lo, hi]+[c[axis]+offset*radius for c, radius in centers for offset in (-1, 0, 1)
                                  if lo < c[axis]+offset*radius < hi]
            cuts = []
            for value in sorted(candidates):
                if not cuts or value-cuts[-1] > 32*np.finfo(float).eps*max(1., abs(lo), abs(hi)):
                    cuts.append(value)
            axis_points, axis_weights = [], []
            for start, stop in zip(cuts[:-1], cuts[1:]):
                count = max(int(np.ceil(nodes*(stop-start)/(hi-lo))), support_nodes if centers else nodes)
                z, w = gauss(count)
                axis_points.append(start+(stop-start)*z)
                axis_weights.append((stop-start)*w)
            axes.append((np.concatenate(axis_points), np.concatenate(axis_weights)))
        points.append(np.stack(np.meshgrid(axes[0][0], axes[1][0], indexing="ij"), -1).reshape(-1, 2))
        weights.append(np.outer(axes[0][1], axes[1][1]).reshape(-1))
    size = max(len(row) for row in weights)
    padded_points, padded_weights = [], []
    for cell, xy, w in zip(model.cells, points, weights):
        lo, hi, bottom, top = cell["bounds"]
        padding = np.tile([(lo+hi)/2, (bottom+top)/2], (size-len(w), 1))
        padded_points.append(np.concatenate([xy, padding]))
        padded_weights.append(np.pad(w, (0, size-len(w))))
    _, _, faces, face_weights = quadrature(model.cells, 2, edge_nodes)
    return torch.tensor(np.array(padded_points)), torch.tensor(np.array(padded_weights)), faces, face_weights


def interface_trace_probe(model, xy, support, side, component, radius, mode="constant"):
    """Move one physical-interface trace, with zero values on all other faces."""
    edge = model.edges[support["edge"]]
    cell, face = edge[side], edge["f"+side]
    bounds = model.cells[cell]["bounds"]
    a = (xy[..., 0]-bounds[0])/(bounds[1]-bounds[0])
    b = (xy[..., 1]-bounds[2])/(bounds[3]-bounds[2])
    z = b if edge["axis"] == 0 else a
    extension = (1-a, a, 1-b, b)[face]
    local = (xy-xy.new_tensor(support["endpoints"][0]))/radius
    window = torch.relu(1-local.square().sum(-1))**3
    polynomial = (torch.ones_like(z) if mode == "constant" else
                  local[..., 1-edge["axis"]])
    mask = (torch.arange(len(model.cells), device=xy.device) == cell)[:, None]
    scalar = 4*z*(1-z)*extension*window*polynomial*mask
    unit = xy.new_tensor([1., 0.] if component == 0 else [0., 1.])
    return scalar[..., None]*unit


def interface_readouts(model, supports, n):
    z, w = gauss(n)
    rows = []
    for segment in supports:
        edge = model.edges[segment["edge"]]
        start, stop = np.array(segment["endpoints"])
        points = start+z[:, None]*(stop-start)
        length = float(np.linalg.norm(stop-start))
        a, b = edge["a"], edge["b"]
        pa, pb = sample_one(model, a, points), sample_one(model, b, points)
        jump = pb[:, :2]-pa[:, :2]
        normal, tangent = np.array(edge["normal"]), np.array(edge["tangent"])
        spring = (float(model.k[0])*(jump@normal)[:, None]*normal
                  +float(model.k[1])*(jump@tangent)[:, None]*tangent)
        integrate = lambda values: length*(w[:, None]*values).sum(0)
        resultants = {"a": integrate(traction(pa, edge["axis"])),
                      "b": integrate(traction(pb, edge["axis"])), "spring": integrate(spring)}
        rows.append({**segment, "a_material": model.cells[a]["material"],
                     "b_material": model.cells[b]["material"], "normal": normal.tolist(),
                     "tangent": tangent.tolist(), "length_over_H": length,
                     "resultants_xy": {k: v.tolist() for k, v in resultants.items()},
                     "resultants_nt": {k: [float(v@normal), float(v@tangent)] for k, v in resultants.items()},
                     "mean_jump_nt_over_Delta": [(w*(jump@normal)).sum().item(), (w*(jump@tangent)).sum().item()],
                     "rms_jump_nt_over_Delta": [np.sqrt((w*(jump@normal)**2).sum()).item(),
                                                 np.sqrt((w*(jump@tangent)**2).sum()).item()]})
    return rows


def outer_boundary_readout(model, n):
    _, _, faces, weights = quadrature(model.cells, 2, n)
    with torch.no_grad():
        values = model(faces).reshape(len(model.cells), 4, n, 5).numpy()
    points = faces.reshape(len(model.cells), 4, n, 2).numpy()
    rows, net, moment, work = [], np.zeros(2), 0., 0.
    top, bottom = 0., 0.
    for edge in model.geometry["outer_boundaries"]:
        i, face = edge["cell"], edge["face"]
        normal = np.array(edge["normal"])
        p, xy, w = values[i, face], points[i, face], weights[i, face].numpy()
        t = np.column_stack([p[:, 2]*normal[0]+p[:, 4]*normal[1],
                             p[:, 4]*normal[0]+p[:, 3]*normal[1]])
        force = (w[:, None]*t).sum(0)
        net += force
        moment += float((w*(xy[:, 0]*t[:, 1]-xy[:, 1]*t[:, 0])).sum())
        work += float((w*(t*p[:, :2]).sum(-1)).sum())
        if edge["side"] == "top":
            top -= force[1]
        if edge["side"] == "bottom":
            bottom += force[1]
        rows.append({**edge, "resultant_xy": force.tolist()})
    return {"edges": rows, "top_compression": top, "bottom_compression": bottom,
            "net_force_xy": net.tolist(), "net_moment": moment, "boundary_work": work}


def section_readouts(model, positions, n):
    z, w = gauss(n)
    rows = []
    for y in positions:
        forces = {"soil": 0., "grout": 0.}
        for i, cell in enumerate(model.cells):
            x0, x1, y0, y1 = cell["bounds"]
            if not y0 < y < y1:
                continue
            p = sample_one(model, i, np.column_stack([x0+z*(x1-x0), np.full(n, y)]))
            forces[cell["material"]] -= float((x1-x0)*(w*p[:, 3]).sum())
        total = sum(forces.values())
        rows.append({"y_over_H": y, "x_bounds_over_H": model.geometry["bounds"][:2],
                     "soil_force": forces["soil"], "grout_force": forces["grout"],
                     "total_force": total, "grout_share": forces["grout"]/total if abs(total) > 1e-12 else None})
    return rows


def evaluate_full(model, nodes=24, edge_nodes=64, region_nodes=16, support_nodes=None):
    quad = list(full_domain_quadrature(model, nodes, edge_nodes, support_nodes))
    quad[0].requires_grad_(True)
    residual = energy_residual(model, quad).detach()
    bulk_size = 3*quad[0].shape[0]*quad[0].shape[1]
    bulk = float(.5*residual[:bulk_size].square().sum())
    spring = float(.5*residual[bulk_size:].square().sum())
    supports = observation_supports(model.geometry)
    boundary = outer_boundary_readout(model, edge_nodes)
    case = model.case
    width = model.geometry["bounds"][1]-model.geometry["bounds"][0]
    regions = [{"id": r["id"], **region_mean_strain(model, r["bounds"], region_nodes)}
               for r in supports["soil_regions"]]
    return {"geometry": model.geometry, "observation_supports": supports,
            "case": asdict(case), "quadrature": {"nodes": nodes, "edge_nodes": edge_nodes, "region_nodes": region_nodes,
                                                  "support_nodes": support_nodes,
                                                  "bulk_points_per_cell": (quad[1] > 0).sum(1).tolist()},
            "bulk_energy": bulk, "spring_energy": spring, "stored_energy": bulk+spring,
            "energy_conjugate_full_reaction": 2*(bulk+spring),
            "boundary": boundary,
            "nominal_axial_stiffness_MPa": case.soil_E_MPa*boundary["top_compression"]/width,
            "interface_segments": interface_readouts(model, supports["segments"], edge_nodes),
            "regions": regions, "sections": section_readouts(model, supports["section_y"], edge_nodes),
            "scales": {"length_mm": case.specimen_size_mm, "displacement_mm": case.applied_shortening_mm,
                       "strain": case.applied_shortening_mm/case.specimen_size_mm,
                       "stress_MPa": case.soil_E_MPa*case.applied_shortening_mm/case.specimen_size_mm,
                       "line_resultant_N_per_mm_thickness": case.soil_E_MPa*case.applied_shortening_mm,
                       "energy_N_per_unit_mm_thickness": case.soil_E_MPa*case.applied_shortening_mm**2}}


def full_local_probe(model, xy, center, radius, material, component, mode, part):
    local = (xy-xy.new_tensor(center))/radius
    x, y = local[..., 0], local[..., 1]
    polynomial = {"constant": torch.ones_like(x), "x": x, "y": y, "xy": x*y}[mode]
    mask = xy.new_tensor([c["material"] == material for c in model.cells])[:, None]
    value = torch.relu(1-x*x-y*y)**3*polynomial*mask
    if component == 1:
        value = value*platen_factor(xy[..., 1], model.geometry["bounds"])
    else:
        # Same global rigid datum shift as the trial correction.
        q = np.subtract(model.geometry["horizontal_datum"]["point"], center)/radius
        datum_poly = {"constant": 1., "x": q[0], "y": q[1], "xy": q[0]*q[1]}[mode]
        anchor = max(0., 1-float(q@q))**3*datum_poly*(material == "soil")
        value = value-anchor
    if part != "full":
        b = model.bounds
        a = (xy[..., 0]-b[..., 0])/(b[..., 1]-b[..., 0])
        c = (xy[..., 1]-b[..., 2])/(b[..., 3]-b[..., 2])
        bubble = 16*a*(1-a)*c*(1-c)
        value = value*(bubble if part == "bubble" else 1-bubble)
    return value[..., None]*xy.new_tensor([1., 0.] if component == 0 else [0., 1.])


def full_probe_work(model, nodes=24, edge_nodes=64, radius=None, support_nodes=None):
    radius = model.vein_width if radius is None else radius
    quad = list(full_domain_quadrature(model, nodes, edge_nodes, support_nodes))
    quad[0].requires_grad_(True)
    field = energy_residual(model, quad).detach()
    bulk_size = 3*quad[0].shape[0]*quad[0].shape[1]
    rows = []
    for corner in model.geometry["corners"]:
        if corner["role"] != "junction":
            continue
        for material in ("soil", "grout"):
            for component in (0, 1):
                for mode in ("constant", "x", "y", "xy"):
                    for part in ("full", "bubble", "trace"):
                        probe = lambda xy: full_local_probe(model, xy, corner["point"], radius,
                                                            material, component, mode, part)
                        vector = displacement_residual(model, probe, quad).detach()
                        bulk = float(field[:bulk_size]@vector[:bulk_size])
                        spring = float(field[bulk_size:]@vector[bulk_size:])
                        norm = float(vector.norm())
                        rows.append({"corner_id": corner["id"], "material": material,
                                     "component": "xy"[component], "mode": mode, "part": part,
                                     "bulk_work": bulk, "spring_work": spring, "virtual_work": bulk+spring,
                                     "self_stiffness": norm**2, "work_over_energy_norm": (bulk+spring)/norm})
    return {"radius_over_H": radius, "nodes": nodes, "edge_nodes": edge_nodes,
            "support_nodes": support_nodes, "probes": rows}
