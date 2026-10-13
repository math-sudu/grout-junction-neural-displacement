"""Two-dimensional forward, plane-strain PINN for embedded grout veins.

Rectangular patches resolve finite ends and a connected cross junction. Real
soil/grout boundaries have independent displacement traces and elastic springs;
artificial cuts within one material remain perfectly bonded. All inputs are
prescribed. Symmetry uses one quarter of the specimen. No observations are fit.
CSV coordinates use H, displacements use Delta, stresses use Es*Delta/H,
and strains use Delta/H as their respective scales. Shear strain is engineering
shear. Area and line quadrature weights use H squared and H.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import linalg
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Case:
    geometry: str = "straight"
    soil_E_MPa: float = 79.9
    grout_E_MPa: float = 12700.
    poisson: float = .3
    specimen_size_mm: float = 100.
    applied_shortening_mm: float = .1
    normal_K_MPa_per_mm: float = 10.
    tangent_K_MPa_per_mm: float = 10.
    grout_area_fraction: float = .1
    vertical_span: float = .8
    horizontal_span: float = .6


def geometry(case):
    """Return a conforming rectangular partition of [0,.5]^2."""
    if case.geometry == "straight":
        width = case.grout_area_fraction / case.vertical_span
        xs, ys = [0., width/2, .5], [0., case.vertical_span/2, .5]
    elif case.geometry == "cross":
        span = case.vertical_span + case.horizontal_span
        width = (span-np.sqrt(span**2-4*case.grout_area_fraction))/2
        xs = [0., width/2, case.horizontal_span/2, .5]
        ys = [0., width/2, case.vertical_span/2, .5]
    else:
        raise ValueError(case.geometry)
    cells = []
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            grout = (i == 0 and ys[j+1] <= case.vertical_span/2)
            if case.geometry == "cross":
                grout |= j == 0 and xs[i+1] <= case.horizontal_span/2
            cells.append({"bounds": [xs[i], xs[i+1], ys[j], ys[j+1]],
                          "material": "grout" if grout else "soil"})
    edges = []
    nx, ny = len(xs)-1, len(ys)-1
    for j in range(ny):
        for i in range(nx):
            a = j*nx+i
            for b, axis, fa, fb in [(a+1, 0, 1, 0), (a+nx, 1, 3, 2)]:
                if (axis == 0 and i == nx-1) or (axis == 1 and j == ny-1):
                    continue
                edges.append({"a": a, "b": b, "axis": axis, "fa": fa, "fb": fb,
                              "interface": cells[a]["material"] != cells[b]["material"]})
    return cells, edges, width


def gauss(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x+1)/2, w/2


def quadrature(cells, n, edge_n):
    z, w = gauss(n)
    ze, we = gauss(edge_n)
    zz = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
    ww = np.outer(w, w).reshape(-1)
    points, weights, faces, face_weights = [], [], [], []
    for c in cells:
        x0, x1, y0, y1 = c["bounds"]
        dx, dy = x1-x0, y1-y0
        points.append(zz * [dx, dy] + [x0, y0])
        weights.append(ww*dx*dy)
        faces.append(np.concatenate([
            np.column_stack([np.full(edge_n, x0), y0+ze*dy]),
            np.column_stack([np.full(edge_n, x1), y0+ze*dy]),
            np.column_stack([x0+ze*dx, np.full(edge_n, y0)]),
            np.column_stack([x0+ze*dx, np.full(edge_n, y1)])]))
        face_weights.append(np.stack([we*dy, we*dy, we*dx, we*dx]))
    return tuple(torch.tensor(np.array(a)) for a in (points, weights, faces, face_weights))


class SharedTrace(nn.Module):
    """Coons displacement patches with shared traces within each material.

Separate material keys retain the two traces of physical interfaces. Edge
corrections vanish at their endpoints; a common vertex value closes junctions.
    """
    def __init__(self, cells, case, scaling, full_domain=False, edge_width=12):
        super().__init__()
        vertices, vertex_keys, corners, edge_keys, edge_ids = [], {}, [], {}, []
        for c in cells:
            x0, x1, y0, y1 = c["bounds"]
            ids = []
            for x, y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
                key = (c["material"], x, y)
                if key not in vertex_keys:
                    vertex_keys[key] = len(vertices)
                    vertices.append(key)
                ids.append(vertex_keys[key])
            corners.append(ids)
            face_ids = []
            for a, b in [(0, 2), (1, 3), (0, 1), (2, 3)]:
                key = (ids[a], ids[b])
                if key not in edge_keys:
                    edge_keys[key] = len(edge_keys)
                face_ids.append(edge_keys[key])
            edge_ids.append(face_ids)
        self.register_buffer("corners", torch.tensor(corners))
        self.register_buffer("edge_ids", torch.tensor(edge_ids))
        initial, mask, fixed = [], [], []
        xmin = min(c["bounds"][0] for c in cells)
        ymin = min(c["bounds"][2] for c in cells)
        ymax = max(c["bounds"][3] for c in cells)
        for material, x, y in vertices:
            e = case.grout_E_MPa/case.soil_E_MPa if material == "grout" else 1.
            if full_domain:
                initial.append([case.poisson/(1-case.poisson)*(x-xmin)/e, -y/e])
                mask.append([not (x == xmin and y == ymin), y not in (ymin, ymax)])
                fixed.append([0., .5 if y == ymin else -.5 if y == ymax else 0.])
            else:
                initial.append([case.poisson/(1-case.poisson)*x/e, -y/e])
                mask.append([x != 0., y not in (0., .5)])
                fixed.append([0., -.5 if y == .5 else 0.])
        node_scale = np.ones(len(vertices))
        edge_scale = np.ones(len(edge_keys))
        if scaling == "compliance":
            node_scale[:], edge_scale[:] = np.inf, np.inf
            for c, ids, faces in zip(cells, corners, edge_ids):
                x0, x1, y0, y1 = c["bounds"]
                e = case.grout_E_MPa/case.soil_E_MPa if c["material"] == "grout" else 1.
                scale = min(x1-x0, y1-y0)/np.sqrt(e)
                for i in ids:
                    node_scale[i] = min(node_scale[i], scale)
                for i in faces:
                    edge_scale[i] = min(edge_scale[i], 4*scale)
        self.node_scale = torch.tensor(node_scale)[:, None]
        self.edge_scale = torch.tensor(edge_scale)[:, None, None]
        self.node_values = nn.Parameter(torch.tensor(initial)/self.node_scale)
        self.register_buffer("node_mask", torch.tensor(mask, dtype=torch.get_default_dtype()))
        self.register_buffer("node_fixed", torch.tensor(fixed))
        masks = []
        for a, b in edge_keys:
            _, xa, ya = vertices[a]
            _, xb, yb = vertices[b]
            masks.append([full_domain or not (xa == xb == 0.),
                          not (ya == yb and ya in ((ymin, ymax) if full_domain else (0., .5)))])
        self.register_buffer("edge_mask", torch.tensor(masks, dtype=torch.get_default_dtype())[:, None, :])
        ne = len(edge_keys)
        self.w1 = nn.Parameter(torch.randn(ne, 1, edge_width)*.5)
        self.b1 = nn.Parameter(torch.zeros(ne, 1, edge_width))
        self.w2 = nn.Parameter(torch.zeros(ne, edge_width, 2))
        self.b2 = nn.Parameter(torch.zeros(ne, 1, 2))

    def correction(self, z, face):
        ids = self.edge_ids[:, face]
        h = torch.tanh(torch.bmm(2*z-1, self.w1[ids])+self.b1[ids])
        return z*(1-z)*(torch.bmm(h, self.w2[ids])+self.b2[ids])*self.edge_mask[ids]*self.edge_scale[ids]

    def forward(self, a, b, interior, homogeneous=False):
        nodes = (self.node_values*self.node_scale*self.node_mask
                 +(0. if homogeneous else self.node_fixed))[self.corners]
        bilinear = ((1-a)*(1-b)*nodes[:, None, 0]+a*(1-b)*nodes[:, None, 1]
                    +(1-a)*b*nodes[:, None, 2]+a*b*nodes[:, None, 3])
        edges = ((1-a)*self.correction(b, 0)+a*self.correction(b, 1)
                 +(1-b)*self.correction(a, 2)+b*self.correction(a, 3))
        return bilinear+edges+16*a*(1-a)*b*(1-b)*interior


def elastic_corner_basis(x, y, theta, length, e, nu):
    """Compatible u/sigma fields from the biharmonic r^2 log(r), r^2 theta modes.

The caller selects the one-sided angular branch from the material sector.
Coordinates are relative to the vertex; the vertex itself has no stress value.
    """
    r2 = x.square()+y.square()
    log = .5*torch.log(r2/length**2)
    c, s = (x.square()-y.square())/r2, 2*x*y/r2
    a = (1+nu)/e
    ulog = torch.stack([a*(2*(1-2*nu)*x*log-x-4*(1-nu)*y*theta),
                        a*(2*(1-2*nu)*y*log-y+4*(1-nu)*x*theta)], -1)
    utheta = torch.stack([2*a*((1-2*nu)*x*theta+2*(1-nu)*y*log),
                          2*a*((1-2*nu)*y*theta-2*(1-nu)*x*log)], -1)
    slog = torch.stack([2*log+2-c, 2*log+2+c, -s], -1)
    stheta = torch.stack([2*theta+s, 2*theta-s, -c], -1)
    return torch.stack([ulog, utheta], -2), torch.stack([slog, stheta], -2)


class PatchNet(nn.Module):
    """Independent patch networks, vectorized over the patch dimension."""
    def __init__(self, case, width=24, seed=7, kinematics="soft", displacement_scaling="unit", corner_enrichment=False, elastic_corners=False):
        super().__init__()
        torch.manual_seed(seed)
        self.case = case
        self.cells, self.edges, self.vein_width = geometry(case)
        self.width = width
        nc = len(self.cells)
        bounds = torch.tensor([c["bounds"] for c in self.cells])
        self.register_buffer("bounds", bounds[:, None, :])
        es = [case.grout_E_MPa/case.soil_E_MPa if c["material"] == "grout" else 1.
              for c in self.cells]
        self.register_buffer("E", torch.tensor(es)[:, None, None])
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n0, n1 in zip([2, width, width], [width, width, 5]):
            a = torch.empty(nc, n0, n1)
            for p in a:
                nn.init.xavier_normal_(p)
            self.weights.append(nn.Parameter(a))
            self.biases.append(nn.Parameter(torch.zeros(nc, 1, n1)))
        nn.init.zeros_(self.weights[-1])
        self.k = torch.tensor([case.normal_K_MPa_per_mm,
                               case.tangent_K_MPa_per_mm])*case.specimen_size_mm/case.soil_E_MPa
        self.kinematics = kinematics
        self.displacement_scaling = displacement_scaling
        self.displacement_scale = torch.ones(nc, 1, 1)
        if displacement_scaling == "compliance":
            self.displacement_scale = torch.minimum(bounds[:, 1]-bounds[:, 0], bounds[:, 3]-bounds[:, 2])[:, None, None]/self.E.sqrt()
        if kinematics == "shared_trace":
            self.trace = SharedTrace(self.cells, case, displacement_scaling)
        self.corner_enrichment = corner_enrichment or elastic_corners
        self.elastic_corners = elastic_corners
        if self.corner_enrichment:
            a = self.vein_width/2
            corners = [(a, case.vertical_span/2)]
            if case.geometry == "cross":
                corners += [(a, a), (case.horizontal_span/2, a)]
            self.singular_corners = torch.tensor(corners)
            self.material_ids = torch.tensor([int(c["material"] == "grout") for c in self.cells])
            if elastic_corners:
                self.elastic_corner_coeff = nn.Parameter(torch.zeros(2, len(corners), 2))
                self.corner_radius = torch.tensor([min(self.vein_width, .5-x, .5-y) for x, y in corners])
                self.reentrant = torch.tensor([False, True, False] if case.geometry == "cross" else [False])
            else:
                self.stress_corner_coeff = nn.Parameter(torch.zeros(nc, len(corners)*3, 3))
                self.displacement_corner_coeff = nn.Parameter(torch.zeros(2, len(corners), 4))

    def elastic_basis(self, xy):
        uv = torch.zeros((*xy.shape[:2], len(self.singular_corners), 2, 2))
        stress = torch.zeros((*xy.shape[:2], len(self.singular_corners), 2, 3))
        grout = self.material_ids[:, None, None].bool()
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x = sx*xy[..., 0:1]-self.singular_corners[:, 0]
            y = sy*xy[..., 1:2]-self.singular_corners[:, 1]
            theta = torch.atan2(y, x)
            # Convex grout sectors use (-pi,-pi/2); the reentrant sector uses
            # (pi/2,2pi). These branches agree across artificial grout cuts.
            theta = torch.where(grout & ~self.reentrant & (theta > 0), theta-2*np.pi, theta)
            theta = torch.where(grout & self.reentrant & (theta <= 0), theta+2*np.pi, theta)
            ub, sb = elastic_corner_basis(x, y, theta, self.vein_width, self.E, self.case.poisson)
            window = torch.relu(1-(x.square()+y.square())/self.corner_radius**2)**3
            v = ub*window[..., None, None]
            s = sb*window[..., None, None]
            uv = uv+v*torch.tensor([sx, sy])
            stress = stress+s*torch.tensor([1., 1., sx*sy])
        return torch.cat([uv, stress], -1).flatten(-3, -2)

    def contract_elastic_basis(self, basis):
        coeff = self.elastic_corner_coeff[self.material_ids].flatten(1)[:, None, :, None]
        return (basis*coeff).sum(-2)

    def elastic_corner_fields(self, xy):
        values = self.contract_elastic_basis(self.elastic_basis(xy))
        return values[..., :2], values[..., 2:]

    def prepare_corner_derivatives(self, xy):
        # Quadrature and geometry stay fixed during optimization. Only the
        # coefficients change, so the spatial basis derivatives are evaluated once.
        basis = self.elastic_basis(xy)
        derivatives = []
        nk = basis.shape[-2]
        for k in range(nk):
            dk = [torch.autograd.grad(basis[:, :, k, j].sum(), xy,
                                     retain_graph=not (k == nk-1 and j == 4))[0].detach()
                  for j in range(5)]
            derivatives.append(torch.stack(dk, -2))
        return basis.detach(), torch.stack(derivatives, 2)

    def corner_fields(self, xy):
        if self.elastic_corners:
            return self.elastic_corner_fields(xy)
        # No collocation/evaluation point is the sharp vertex itself. The
        # angular stress limit at that vertex is not a single-valued quantity.
        dx = xy[..., 0:1]-self.singular_corners[:, 0]
        dy = xy[..., 1:2]-self.singular_corners[:, 1]
        r2 = dx.square()+dy.square()
        log = .5*torch.log(r2/self.vein_width**2)
        basis = torch.stack([log, (dx.square()-dy.square())/r2, 2*dx*dy/r2], -1).flatten(-2)
        stress = torch.bmm(basis, self.stress_corner_coeff)
        # Material-shared displacement enrichment keeps every artificial cut
        # continuous. Reflections impose the physical x/y symmetry parities.
        terms = torch.zeros((*xy.shape[:2], len(self.singular_corners), 4))
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            rx = xy[..., 0:1]-sx*self.singular_corners[:, 0]
            ry = xy[..., 1:2]-sy*self.singular_corners[:, 1]
            lr = .5*torch.log((rx.square()+ry.square())/self.vein_width**2)
            terms = terms+torch.stack([rx*lr, sx*sy*ry*lr, ry*lr, sx*sy*rx*lr], -1)
        terms = terms*self.displacement_corner_coeff[self.material_ids][:, None]
        u = terms[..., :2].sum((-1, -2))[..., None]/self.E.sqrt()
        v = terms[..., 2:].sum((-1, -2))[..., None]/self.E.sqrt()*(1-4*xy[..., 1:2]**2)
        return torch.cat([u, v], -1), stress

    def forward(self, xy, include_corners=True, corner_basis=None, homogeneous=False,
                displacement_only=False):
        b = self.bounds
        x, y = xy[..., 0:1], xy[..., 1:2]
        dx, dy = b[..., 1:2]-b[..., 0:1], b[..., 3:4]-b[..., 2:3]
        q = torch.cat([2*(x-b[..., 0:1])/dx-1, 2*(y-b[..., 2:3])/dy-1], -1)
        for i, (w, bias) in enumerate(zip(self.weights, self.biases)):
            q = torch.bmm(q, w)+bias
            if i < 2:
                q = torch.tanh(q)
        # Hard roller/symmetry conditions. Only patches touching a boundary
        # receive its factor; there is no imposed continuity at interior edges.
        fx0 = torch.where(b[..., 0:1] == 0, x/dx, torch.ones_like(x))
        fx1 = torch.where(b[..., 1:2] == .5, (.5-x)/dx, torch.ones_like(x))
        fy0 = torch.where(b[..., 2:3] == 0, y/dy, torch.ones_like(y))
        fy1 = torch.where(b[..., 3:4] == .5, (.5-y)/dy, torch.ones_like(y))
        nu = self.case.poisson
        uq = q[..., :2]*self.displacement_scale
        affine = 0. if homogeneous else 1.
        u = affine*nu/(1-nu)*x/self.E + fx0*uq[..., 0:1]
        v = -affine*y/self.E + fy0*fy1*uq[..., 1:2]
        if self.kinematics == "shared_trace":
            uv = self.trace((x-b[..., 0:1])/dx, (y-b[..., 2:3])/dy, uq, homogeneous)
            u, v = uv[..., :1], uv[..., 1:2]
        if self.corner_enrichment and include_corners:
            if corner_basis is None:
                uc, sc = self.corner_fields(xy)
            else:
                values = self.contract_elastic_basis(corner_basis)
                uc, sc = values[..., :2], values[..., 2:]
            u, v = u+uc[..., :1], v+uc[..., 1:2]
            if not self.elastic_corners and not displacement_only:
                q = torch.cat([q[..., :2], q[..., 2:]+sc], -1)
        if displacement_only:
            return torch.cat([u, v], -1)
        sxx = fx1*q[..., 2:3]
        syy = -affine/(1-nu**2)+q[..., 3:4]
        sxy = fx0*fx1*fy0*fy1*q[..., 4:5]
        result = torch.cat([u, v, sxx, syy, sxy], -1)
        if self.elastic_corners and include_corners:
            # Compact supports vanish at the physical free/platen boundaries;
            # reflected shear modes vanish on both symmetry axes.
            result = torch.cat([result[..., :2], result[..., 2:]+sc], -1)
        return result

    def fields(self, xy, corner_data=None, homogeneous=False):
        p = self(xy, include_corners=corner_data is None, homogeneous=homogeneous)
        d = [torch.autograd.grad(p[..., i].sum(), xy, create_graph=True)[0]
             for i in range(5)]
        if corner_data is not None:
            basis, derivative = corner_data
            p = p+self.contract_elastic_basis(basis)
            coeff = self.elastic_corner_coeff[self.material_ids].flatten(1)[:, None, :, None]
            d = [v+(derivative[..., i, :]*coeff).sum(2) for i, v in enumerate(d)]
        strain = torch.stack([d[0][..., 0], d[1][..., 1],
                              d[0][..., 1]+d[1][..., 0]], -1)
        div = torch.stack([d[2][..., 0]+d[4][..., 1],
                           d[4][..., 0]+d[3][..., 1]], -1)
        nu = self.case.poisson
        compliance = (1+nu)/self.E * torch.stack([
            (1-nu)*p[..., 2]-nu*p[..., 3],
            (1-nu)*p[..., 3]-nu*p[..., 2], 2*p[..., 4]], -1)
        return p, strain, compliance, div


def reparameterize(model, scaling, seed):
    """Change optimizer coordinates while preserving every physical field."""
    new = PatchNet(model.case, model.width, seed, model.kinematics, scaling, model.corner_enrichment, model.elastic_corners)
    new.load_state_dict(model.state_dict())
    with torch.no_grad():
        ratio = model.displacement_scale/new.displacement_scale
        new.weights[-1][..., :2] *= ratio
        new.biases[-1][..., :2] *= ratio
        if model.kinematics == "shared_trace":
            new.trace.node_values *= model.trace.node_scale/new.trace.node_scale
            eratio = model.trace.edge_scale/new.trace.edge_scale
            new.trace.w2 *= eratio
            new.trace.b2 *= eratio
    return new


def enrich_corners(model, seed, elastic=False):
    new = PatchNet(model.case, model.width, seed, model.kinematics, model.displacement_scaling, True, elastic)
    keys = new.load_state_dict(model.state_dict(), strict=False)
    expected = {"elastic_corner_coeff"} if elastic else {"stress_corner_coeff", "displacement_corner_coeff"}
    assert set(keys.missing_keys) == expected
    assert not keys.unexpected_keys
    return new


def traction(p, axis):
    # The same a->b normal is used on both sides of an interior edge.
    return p[..., [2, 4]] if axis == 0 else p[..., [4, 3]]


def edge_constraints(model, pe, wf):
    """Signed trace constraints with the original penalty scaling.

Each row represents one vector constraint on one open edge. The multiplier
uses the same line-quadrature inner product as its quadratic penalty.
    """
    residuals, weights = [], []
    for e in model.edges:
        a, b, ax, fa, fb = [e[k] for k in ("a", "b", "axis", "fa", "fb")]
        pa, pb = pe[a, fa], pe[b, fb]
        ta, tb = traction(pa, ax), traction(pb, ax)
        jump = pb[..., :2]-pa[..., :2]
        residuals.append(ta-tb)
        weights.append(wf[a, fa]/.25)
        if e["interface"]:
            k = model.k if ax == 0 else model.k.flip(0)
            residuals.append(((ta+tb)/2-k*jump)/(1+k).sqrt())
            weights.append(wf[a, fa]/.25)
        elif model.kinematics == "soft":
            eh = 2/(1/model.E[a, 0, 0]+1/model.E[b, 0, 0])
            residuals.append(eh.sqrt()*jump)
            weights.append(wf[a, fa]/.25)
    return torch.stack(residuals), torch.stack(weights)[..., None]


def loss_terms(model, quad, corner_cache=None, dual=None):
    xy, w, face, wf = quad
    _, strain, compliance, div = model.fields(xy, corner_cache["bulk"] if corner_cache else None)
    err = strain-compliance
    nu = model.case.poisson
    # (epsilon-S:sigma):C:(epsilon-S:sigma), in engineering-shear notation.
    lam = model.E[..., 0]*nu/((1+nu)*(1-2*nu))
    mu = model.E[..., 0]/(2*(1+nu))
    constit = lam*err[..., :2].sum(-1)**2 + 2*mu*(err[..., 0]**2+err[..., 1]**2) + mu*err[..., 2]**2
    bulk = (w*(constit+div.square().sum(-1))).sum()/.25
    pe = model(face, corner_basis=corner_cache["edges"] if corner_cache else None).reshape(len(model.cells), 4, -1, 5)
    residual, weight = edge_constraints(model, pe, wf)
    terms = {"bulk": bulk, "edges": (weight*residual.square()).sum()}
    if dual is not None:
        terms["dual_work"] = (weight*dual*residual).sum()
    return terms


def conjugate_gradient(matvec, rhs, steps):
    """Solve a consistent positive-semidefinite system from a zero increment."""
    x = torch.zeros_like(rhs)
    residual, direction = rhs.clone(), rhs.clone()
    squared = residual.dot(residual)
    iterations = 0
    for _ in range(steps):
        if float(squared) == 0.:
            break
        product = matvec(direction)
        curvature = direction.dot(product)
        if not torch.isfinite(curvature) or float(curvature) <= 0:
            raise FloatingPointError("Nonpositive or nonfinite readout curvature")
        alpha = squared/curvature
        x += alpha*direction
        residual -= alpha*product
        new_squared = residual.dot(residual)
        direction = residual+(new_squared/squared)*direction
        squared = new_squared
        iterations += 1
    return x, iterations


def readout_parameters(model):
    linear = [model.weights[-1], model.biases[-1]]
    if model.kinematics == "shared_trace":
        linear += [model.trace.node_values, model.trace.w2, model.trace.b2]
    if model.elastic_corners:
        linear += [model.elastic_corner_coeff]
    elif model.corner_enrichment:
        linear += [model.stress_corner_coeff, model.displacement_corner_coeff]
    return linear


def weighted_residual(model, quad, corner_cache=None, dual=None, homogeneous=False):
    """Square-root form of the mixed objective, including the dual shift.

Its squared norm equals loss_terms plus ||sqrt(W)*dual/2|| squared.
Trace/deviatoric factors preserve the engineering-shear constitutive metric.
    """
    xy, w, face, wf = quad
    _, strain, compliance, div = model.fields(xy, corner_cache["bulk"] if corner_cache else None, homogeneous)
    err = strain-compliance
    nu = model.case.poisson
    lam = model.E[..., 0]*nu/((1+nu)*(1-2*nu))
    mu = model.E[..., 0]/(2*(1+nu))
    constit = torch.stack([(lam+mu).sqrt()*err[..., :2].sum(-1),
                          mu.sqrt()*(err[..., 0]-err[..., 1]),
                          mu.sqrt()*err[..., 2]], -1)
    bulk = torch.cat([constit, div], -1)*(w/.25).sqrt()[..., None]
    pe = model(face, corner_basis=corner_cache["edges"] if corner_cache else None,
               homogeneous=homogeneous).reshape(len(model.cells), 4, -1, 5)
    residual, weight = edge_constraints(model, pe, wf)
    if dual is not None:
        residual = residual+dual/2
    return torch.cat([bulk.flatten(), (weight.sqrt()*residual).flatten()])


def fit_svd_readout(model, quad, corner_cache, dual, rcond=None, matrix_cache=None):
    """Solve the same fixed-feature objective without forming normal equations.

The residual is affine in these coefficients. Homogeneous unit-coefficient
fields assemble its matrix without subtracting two affine fields. Column
normalization changes solver coordinates only. The numerical-rank cutoff is
reported because near-dependent features can affect the recovered fields.
    """
    linear = readout_parameters(model)
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    original = [p.detach().clone() for p in linear]
    before = float(sum(loss_terms(model, quad, corner_cache, dual).values()).detach())
    start = time.time()
    for p in parameters:
        p.requires_grad_(False)
    try:
        with torch.no_grad():
            for p in linear:
                p.zero_()
        # Keep coordinate differentiation active while coefficients are fixed.
        if matrix_cache:
            matrix = matrix_cache["matrix"]
        else:
            base = weighted_residual(model, quad, corner_cache).detach().numpy()
            matrix = np.empty((len(base), sum(p.numel() for p in linear)), order="F")
            column = 0
            for p in linear:
                flat = p.reshape(-1)
                for j in range(p.numel()):
                    flat[j] = 1.
                    matrix[:, column] = weighted_residual(model, quad, corner_cache,
                                                         homogeneous=True).detach().numpy()
                    flat[j] = 0.
                    column += 1
                print(json.dumps({"readout_columns": column, "total_columns": matrix.shape[1],
                                  "seconds": round(time.time()-start, 1)}), flush=True)
            if matrix_cache is not None:
                matrix_cache["matrix"] = matrix
        target = -weighted_residual(model, quad, corner_cache, dual).detach().numpy()
        scales = np.linalg.norm(matrix, axis=0)
        active = scales > 0.
        scaled = matrix[:, active]/scales[active]
        cutoff = rcond if rcond is not None else np.finfo(matrix.dtype).eps*max(scaled.shape)
        solution, _, rank, singular = linalg.lstsq(scaled, target, cond=cutoff, lapack_driver="gelsd")
        coefficients = np.zeros(matrix.shape[1])
        coefficients[active] = solution/scales[active]
        offset = 0
        with torch.no_grad():
            for p in linear:
                p.copy_(torch.from_numpy(coefficients[offset:offset+p.numel()]).reshape_as(p))
                offset += p.numel()
        actual = weighted_residual(model, quad, corner_cache, dual).detach().numpy()
        predicted = matrix@coefficients-target
        after = float(sum(loss_terms(model, quad, corner_cache, dual).values()).detach())
        return {"solver": "svd", "objective_before": before, "objective_after": after,
                "rows": matrix.shape[0], "columns": matrix.shape[1],
                "active_columns": int(active.sum()), "rank": int(rank), "rcond": float(cutoff),
                "singular_max": float(singular[0]),
                "singular_min_retained": float(singular[rank-1]),
                "retained_condition": float(singular[0]/singular[rank-1]),
                "maximum_coefficient": float(np.abs(coefficients).max()),
                "matrix_field_residual_difference": float(np.linalg.norm(actual-predicted)),
                "elapsed_s": time.time()-start}
    except BaseException:
        with torch.no_grad():
            for p, value in zip(linear, original):
                p.copy_(value)
        raise
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def fit_linear_readout(model, quad, corner_cache, dual, steps):
    """Minimize the fixed-feature quadratic using exact autograd Hessian products.

No material input or hidden feature changes. All mixed output coefficients,
shared trace values and active corner coefficients are optimized together.
    """
    linear = readout_parameters(model)
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    linear_ids = {id(p) for p in linear}
    for p in parameters:
        p.requires_grad_(id(p) in linear_ids)
    try:
        loss = sum(loss_terms(model, quad, corner_cache, dual).values())
        gradient = torch.cat([g.flatten() for g in torch.autograd.grad(loss, linear, create_graph=True)])

        def product(direction):
            values = torch.autograd.grad(gradient, linear, grad_outputs=direction, retain_graph=True)
            return torch.cat([v.flatten() for v in values]).detach()

        increment, iterations = conjugate_gradient(product, -gradient.detach(), steps)
        offset = 0
        with torch.no_grad():
            for p in linear:
                p.add_(increment[offset:offset+p.numel()].reshape_as(p))
                offset += p.numel()
        after = sum(loss_terms(model, quad, corner_cache, dual).values())
        return {"iterations": iterations, "objective_before": float(loss.detach()),
                "objective_after": float(after.detach())}
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def train(model, steps, nodes, edge_nodes, dual_updates=0, dual=None, readout_steps=0,
          readout_solver="cg", readout_rcond=None):
    quad = list(quadrature(model.cells, nodes, edge_nodes))
    quad[0].requires_grad_(True)
    corner_cache = None
    if model.elastic_corners:
        corner_cache = {"bulk": model.prepare_corner_derivatives(quad[0]),
                        "edges": model.elastic_basis(quad[2]).detach()}
    if dual_updates:
        with torch.no_grad():
            pe = model(quad[2]).reshape(len(model.cells), 4, -1, 5)
            residual, _ = edge_constraints(model, pe, quad[3])
        if dual is None:
            dual = torch.zeros_like(residual)
        elif dual.shape != residual.shape:
            raise ValueError("Resume augmented training with the saved edge quadrature size.")
    calls = 0
    start = time.time()
    rounds = []
    # This cache is local to one call with fixed features and quadrature.
    readout_cache = {} if steps == 0 and readout_solver == "svd" else None

    def closure():
        nonlocal calls
        opt.zero_grad()
        terms = loss_terms(model, quad, corner_cache, dual)
        loss = sum(terms.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PINN objective")
        loss.backward()
        calls += 1
        if calls % 250 == 0:
            print(json.dumps({"calls": calls, "seconds": round(time.time()-start, 1),
                              **{k: float(v.detach()) for k, v in terms.items()}}), flush=True)
        return loss

    for update in range(dual_updates or 1):
        # A dual update changes the objective; do not reuse its old L-BFGS history.
        if steps:
            opt = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=steps,
                                    max_eval=int(steps*1.35), history_size=60,
                                    tolerance_grad=1e-10, tolerance_change=1e-13,
                                    line_search_fn="strong_wolfe")
            opt.step(closure)
        if readout_solver == "svd":
            readout = fit_svd_readout(model, quad, corner_cache, dual, readout_rcond, readout_cache)
        else:
            readout = fit_linear_readout(model, quad, corner_cache, dual, readout_steps) if readout_steps else None
        final = loss_terms(model, quad, corner_cache, dual)
        record = {"update": update, "closure_calls": calls,
                  **{k: float(v.detach()) for k, v in final.items()}}
        if readout is not None:
            record["readout"] = readout
        if readout_solver == "svd":
            response = evaluate(model, max(36, nodes))[0]
            record["response"] = {key: response[key] for key in (
                "axial_modulus_MPa", "regions", "interface_resultants",
                "twice_stored_energy_over_boundary_work",
                "internal_virtual_work_over_boundary_work")}
        rounds.append(record)
        if dual_updates:
            with torch.no_grad():
                pe = model(quad[2]).reshape(len(model.cells), 4, -1, 5)
                residual, _ = edge_constraints(model, pe, quad[3])
                # Function-space update for L = bulk + ||r||_W^2 + <lambda,r>_W.
                dual = dual+2*residual
            print(json.dumps({k: v for k, v in record.items() if k != "response"}), flush=True)
    return {"closure_calls": calls, "elapsed_s": time.time()-start,
            "requested_steps": steps, "nodes_per_axis": nodes, "edge_nodes": edge_nodes,
            "readout_steps": readout_steps,
            "readout_solver": readout_solver,
            "interface_solver": "augmented" if dual_updates else "penalty",
            "dual_updates": dual_updates, "rounds": rounds,
            "objective_terms": {k: float(v.detach()) for k, v in final.items()}}, dual


def sample_one(model, cell, points, gradients=False):
    # Evaluation coordinates for other patches do not contribute to this result.
    p = np.array(points)
    arr = np.empty((len(model.cells), len(p), 2))
    for i, c in enumerate(model.cells):
        x0, x1, y0, y1 = c["bounds"]
        arr[i, :, :] = [(x0+x1)/2, (y0+y1)/2]
    arr[cell] = p
    xy = torch.tensor(arr, requires_grad=gradients)
    if gradients:
        f, eps, _, _ = model.fields(xy)
        return f[cell].detach().numpy(), eps[cell].detach().numpy()
    with torch.no_grad():
        return model(xy)[cell].numpy()


def region_mean_strain(model, box, n=24):
    z, w = gauss(n)
    zz = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
    ww = np.outer(w, w).ravel()
    integral, area = np.zeros(3), 0.
    for i, c in enumerate(model.cells):
        if c["material"] != "soil":
            continue
        x0, x1, y0, y1 = c["bounds"]
        x0, x1, y0, y1 = max(x0, box[0]), min(x1, box[1]), max(y0, box[2]), min(y1, box[3])
        if x1 <= x0 or y1 <= y0:
            continue
        _, eps = sample_one(model, i, zz*[x1-x0, y1-y0]+[x0, y0], True)
        a = (x1-x0)*(y1-y0)
        integral += (ww[:, None]*eps).sum(0)*a
        area += a
    return {"bounds_over_size": box, "soil_area": area,
            "mean_compression_over_nominal": float(-integral[1]/area),
            "mean_engineering_shear_over_nominal": float(integral[2]/area)}


def evaluate(model, n=36, edge_nodes=96, region_nodes=24):
    xy, w, face, wf = quadrature(model.cells, n, edge_nodes)
    xy.requires_grad_(True)
    f, eps, compliance, div = model.fields(xy)
    f, eps, compliance, div = [a.detach().numpy() for a in (f, eps, compliance, div)]
    x, w, wf = xy.detach().numpy(), w.numpy(), wf.numpy()
    with torch.no_grad():
        pe = model(face).reshape(len(model.cells), 4, -1, 5).numpy()
    reaction_top, reaction_bottom, outer_work = 0., 0., 0.
    cv = []
    for i, c in enumerate(model.cells):
        if c["bounds"][3] == .5:
            reaction_top -= (wf[i, 3]*pe[i, 3, :, 3]).sum()
        if c["bounds"][2] == 0:
            reaction_bottom -= (wf[i, 2]*pe[i, 2, :, 3]).sum()
        forces = []
        for fa, ax, sign in [(0, 0, -1), (1, 0, 1), (2, 1, -1), (3, 1, 1)]:
            forces.append(sign*(wf[i, fa, :, None]*traction(pe[i, fa], ax)).sum(0))
            if c["bounds"][fa] == (0. if fa in (0, 2) else .5):
                outer_work += sign*(wf[i, fa]*(traction(pe[i, fa], ax)*pe[i, fa, :, :2]).sum(-1)).sum()
        cv.append({"cell": i, "material": c["material"],
                   "face_forces": np.array(forces).tolist(),
                   "net_force": np.sum(forces, axis=0).tolist()})
    edge_rows, interface_resultants, interface_twice = [], [], 0.
    jump_error_integral, law_error_integral, bond_jump_max = 0., 0., 0.
    defect = {"artificial_cut_work": 0., "interface_law_work": 0.,
              "traction_imbalance_work": 0.,
              "bulk_equilibrium_work": float((w*(div*f[..., :2]).sum(-1)).sum())}
    for ie, e in enumerate(model.edges):
        a, b, ax, fa, fb = [e[k] for k in ("a", "b", "axis", "fa", "fb")]
        pa, pb = pe[a, fa], pe[b, fb]
        ta, tb = traction(pa, ax), traction(pb, ax)
        jump = pb[:, :2]-pa[:, :2]
        mean_t = .5*(ta+tb)
        defect["traction_imbalance_work"] -= float((wf[a, fa]*(.5*(ta-tb)*(pa[:, :2]+pb[:, :2])).sum(-1)).sum())
        jump_error_integral += (wf[a, fa]*((ta-tb)**2).sum(-1)).sum()
        if not e["interface"]:
            bond_jump_max = max(bond_jump_max, float(np.linalg.norm(jump, axis=1).max()))
            defect["artificial_cut_work"] += float((wf[a, fa]*(mean_t*jump).sum(-1)).sum())
            continue
        k = model.k.numpy() if ax == 0 else model.k.numpy()[::-1]
        defect["interface_law_work"] += float((wf[a, fa]*((mean_t-k*jump)*jump).sum(-1)).sum())
        interface_twice += (wf[a, fa]*(k*jump**2).sum(-1)).sum()
        law_error_integral += (wf[a, fa]*((.5*(ta+tb)-k*jump)**2).sum(-1)).sum()
        length = wf[a, fa].sum()
        interface_resultants.append({"edge": ie, "normal_axis": ax,
            "a": a, "b": b, "length_over_size": float(length),
            "mean_jump_normal": float((wf[a, fa]*jump[:, ax]).sum()/length),
            "mean_jump_tangent": float((wf[a, fa]*jump[:, 1-ax]).sum()/length),
            "stress_resultant": (wf[a, fa, :, None]*mean_t).sum(0).tolist(),
            "spring_resultant": (wf[a, fa, :, None]*k*jump).sum(0).tolist()})
        coord = face.numpy().reshape(len(model.cells), 4, -1, 2)[a, fa]
        for j in range(len(coord)):
            edge_rows.append({"edge": ie, "a": a, "b": b, "normal_axis": ax,
                              "x": coord[j, 0], "y": coord[j, 1], "weight": wf[a, fa, j],
                              "jump_normal": jump[j, ax], "jump_tangent": jump[j, 1-ax],
                              "traction_normal": .5*(ta[j, ax]+tb[j, ax]),
                              "traction_tangent": .5*(ta[j, 1-ax]+tb[j, 1-ax]),
                              "traction_mismatch_x": ta[j, 0]-tb[j, 0],
                              "traction_mismatch_y": ta[j, 1]-tb[j, 1]})
    stress = f[..., [2, 3, 4]]
    twice_bulk = (w*(stress*compliance).sum(-1)).sum()
    internal_work = (w*(stress*eps).sum(-1)).sum()+interface_twice
    boundary_work = reaction_top*.5
    # Constitutive stresses in a displacement-energy formulation satisfy free
    # tractions weakly. Include their work when checking integration by parts.
    defect["natural_boundary_work"] = float(boundary_work-outer_work)
    section_rows = []
    z, wz = gauss(edge_nodes)
    for y in [0., .01, .025, .075, .15, .25, .35, .39, .41, .45, .49]:
        force, grout_force = 0., 0.
        for i, c in enumerate(model.cells):
            x0, x1, y0, y1 = c["bounds"]
            if not y0 <= y < y1:
                continue
            vals = sample_one(model, i, np.column_stack([x0+(x1-x0)*z, np.full(len(z), y)]))
            q = -(wz*vals[:, 3]).sum()*(x1-x0)
            force += q
            if c["material"] == "grout":
                grout_force += q
        section_rows.append({"y": y, "axial_force_quarter": force,
                             "grout_force_quarter": grout_force, "grout_load_fraction": grout_force/force,
                             "section_force_over_top": force/reaction_top})
    halfw, tip = model.vein_width/2, model.case.vertical_span/2
    regions = {"vertical_tip_cap": region_mean_strain(model, [0., halfw, tip, tip+.05], region_nodes),
               "vertical_tip_side": region_mean_strain(model, [halfw, halfw+.05, tip-.05, tip], region_nodes)}
    if model.case.geometry == "cross":
        regions["junction_shoulder"] = region_mean_strain(model, [halfw, halfw+.05, halfw, halfw+.05], region_nodes)
        ht = model.case.horizontal_span/2
        regions["horizontal_tip_cap"] = region_mean_strain(model, [ht, ht+.05, 0., halfw], region_nodes)
    field_rows = []
    for i, c in enumerate(model.cells):
        for j in range(len(x[i])):
            field_rows.append([i, c["material"], *x[i, j], w[i, j], *f[i, j], *eps[i, j]])
    summary = {"axial_modulus_MPa": float(2*reaction_top*model.case.soil_E_MPa),
               "top_reaction_quarter": float(reaction_top),
               "bottom_over_top_reaction": float(reaction_bottom/reaction_top),
               "twice_stored_energy_over_boundary_work": float((twice_bulk+interface_twice)/boundary_work),
               "internal_virtual_work_over_boundary_work": float(internal_work/boundary_work),
               "virtual_work_defect_terms_over_boundary_work": {k: v/boundary_work for k, v in defect.items()},
               "bulk_twice_energy": float(twice_bulk), "interface_twice_energy": float(interface_twice),
               "max_patch_net_force_over_top": float(max(np.linalg.norm(a["net_force"]) for a in cv)/reaction_top),
               "traction_jump_L2_over_top": float(np.sqrt(jump_error_integral)/reaction_top),
               "interface_law_L2_over_top": float(np.sqrt(law_error_integral)/reaction_top),
               "maximum_artificial_cut_displacement_jump": bond_jump_max,
               "regions": regions, "patch_balances": cv,
               "interface_resultants": interface_resultants,
               "section_force_range_over_top": [float(min(r["section_force_over_top"] for r in section_rows)),
                                                float(max(r["section_force_over_top"] for r in section_rows))]}
    fields = pd.DataFrame(field_rows, columns=["cell", "material", "x", "y", "area_weight", "ux", "uy", "sxx", "syy", "sxy", "exx", "eyy", "gamma_xy"])
    return summary, fields, pd.DataFrame(edge_rows), pd.DataFrame(section_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["straight", "cross"], default="straight")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--nodes", type=int, default=12)
    ap.add_argument("--edge-nodes", type=int, default=32)
    ap.add_argument("--dual-updates", type=int, default=0,
                    help="Augmented-Lagrangian updates; --steps is the inner solve length.")
    ap.add_argument("--readout-steps", type=int, default=0,
                    help="Conjugate-gradient steps for the fixed-feature quadratic after each inner solve.")
    ap.add_argument("--readout-solver", choices=["cg", "svd"], default="cg")
    ap.add_argument("--readout-rcond", type=float,
                    help="SVD relative rank cutoff; default is machine epsilon times the larger matrix dimension.")
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--normal-k", type=float, default=10.)
    ap.add_argument("--tangent-k", type=float, default=10.)
    ap.add_argument("--kinematics", choices=["soft", "shared_trace"], default="shared_trace")
    ap.add_argument("--displacement-scaling", choices=["unit", "compliance"])
    corner_group = ap.add_mutually_exclusive_group()
    corner_group.add_argument("--corner-enrichment", action="store_true")
    corner_group.add_argument("--elastic-corners", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    seed, dual = args.seed, None
    if args.resume:
        saved = torch.load(args.resume, weights_only=False)
        case = Case(**saved["case"])
        seed = saved["seed"]
        model = PatchNet(case, saved["width"], saved["seed"], saved.get("kinematics", "soft"), saved.get("displacement_scaling", "unit"), saved.get("corner_enrichment", False), saved.get("elastic_corners", False))
        model.load_state_dict(saved["state_dict"])
        if args.dual_updates:
            dual = saved.get("interface_dual")
        if (args.elastic_corners and model.corner_enrichment and not model.elastic_corners) or (args.corner_enrichment and model.elastic_corners):
            raise ValueError("Start a different corner variant from a checkpoint without corner enrichment.")
        if args.displacement_scaling and args.displacement_scaling != model.displacement_scaling:
            model = reparameterize(model, args.displacement_scaling, seed)
        if (args.corner_enrichment or args.elastic_corners) and not model.corner_enrichment:
            model = enrich_corners(model, seed, args.elastic_corners)
    else:
        case = Case(geometry=args.geometry, normal_K_MPa_per_mm=args.normal_k,
                    tangent_K_MPa_per_mm=args.tangent_k)
        model = PatchNet(case, args.width, args.seed, args.kinematics, args.displacement_scaling or "compliance", args.corner_enrichment, args.elastic_corners)
    info, dual = train(model, args.steps, args.nodes, args.edge_nodes,
                       args.dual_updates, dual, args.readout_steps,
                       args.readout_solver, args.readout_rcond) if args.steps or args.readout_steps or args.readout_solver == "svd" else ({}, dual)
    summary, fields, edges, sections = evaluate(model, max(36, args.nodes))
    out = ROOT/"results/feasibility/partitioned_2d"
    out.mkdir(parents=True, exist_ok=True)
    stem = out/args.name
    fields.to_csv(stem.with_suffix(".csv"), index=False)
    edges.to_csv(stem.with_name(stem.name+"_interfaces.csv"), index=False)
    sections.to_csv(stem.with_name(stem.name+"_sections.csv"), index=False)
    payload = {"case": asdict(case), "plane_condition": "plane_strain", "width": model.width,
               "seed": seed, "cells": model.cells, "edges": model.edges,
               "kinematics": model.kinematics,
               "displacement_scaling": model.displacement_scaling,
               "corner_enrichment": model.corner_enrichment,
               "elastic_corners": model.elastic_corners,
               "vein_width_over_size": model.vein_width, "optimization": info,
               "parent_checkpoint": str(args.resume) if args.resume else None,
               "results": summary}
    stem.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "case": asdict(case),
                "width": model.width, "seed": seed, "kinematics": model.kinematics,
                "displacement_scaling": model.displacement_scaling,
                "corner_enrichment": model.corner_enrichment, "elastic_corners": model.elastic_corners,
                "interface_dual": dual}, stem.with_suffix(".pt"))
    print(json.dumps({"name": args.name, **{k: v for k, v in summary.items() if k != "patch_balances"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
