"""Finite-scale equilibrium and traction constraints on neural displacement.

The operator integrates constitutive stress and spring displacement jumps.
It introduces no stress unknowns or observed force targets. Cell force balance
uses the divergence theorem; face equations use Legendre moments. The operator
is intended for the existing fixed-feature affine coefficient solve.
"""
from __future__ import annotations

from collections import Counter

# Initialize the shared numerical module before loading additional native
# extensions, as in the original solver entry point on Windows.
from partitioned_mixed_pinn import gauss

import numpy as np
from scipy import linalg, sparse
import torch

from local_virtual_work import strain_of


def distinct_cuts(values):
    cuts = []
    for value in sorted(values):
        if not cuts or value-cuts[-1] > 1e-12:
            cuts.append(value)
    return cuts


class EquationMoments:
    def __init__(self, model, edge_nodes=64, degree=2, junction_radius_over_arm_width=.5):
        self.model = model
        self.edge_nodes, self.degree = edge_nodes, degree
        radius = junction_radius_over_arm_width*model.vein_width
        if edge_nodes <= degree or degree < 0 or radius <= 0:
            raise ValueError("Moment quadrature and junction radius must be positive and resolved.")
        centers = [c["point"] for c in model.geometry["corners"] if c["role"] == "junction"]
        z, w = gauss(edge_nodes)
        legendre = np.polynomial.legendre.legvander(2*z-1, degree)
        points = [[] for _ in model.cells]
        point_count = [0 for _ in model.cells]
        row_ids, cell_ids, sample_ids, field_ids, factors = [], [], [], [], []
        lines = {}
        self.rows = []

        def line(cell, axis, position, lo, hi):
            key = (cell, axis, position, lo, hi)
            if key not in lines:
                xy = np.empty((edge_nodes, 2))
                xy[:, axis], xy[:, 1-axis] = position, lo+(hi-lo)*z
                ids = np.arange(point_count[cell], point_count[cell]+edge_nodes)
                points[cell].append(xy)
                point_count[cell] += edge_nodes
                lines[key] = ids
            return lines[key]

        def row(**metadata):
            self.rows.append(metadata)
            return len(self.rows)-1

        def term(index, cell, samples, component, weights):
            row_ids.extend([index]*len(samples))
            cell_ids.extend([cell]*len(samples))
            sample_ids.extend(samples)
            field_ids.extend([component]*len(samples))
            factors.extend(weights)

        # Constant-vector tests of div(sigma)=0 on each rectangle. Refinement
        # follows the half-arm-width junction neighborhood on both materials.
        for cell_id, cell in enumerate(model.cells):
            a, b, c, d = cell["bounds"]
            nearby = [(x, y) for x, y in centers
                      if x+radius > a and x-radius < b and y+radius > c and y-radius < d]
            xs = distinct_cuts([a, b]+[x+k*radius for x, _ in nearby for k in (-1, 0, 1)
                                       if a < x+k*radius < b])
            ys = distinct_cuts([c, d]+[y+k*radius for _, y in nearby for k in (-1, 0, 1)
                                       if c < y+k*radius < d])
            for x0, x1 in zip(xs[:-1], xs[1:]):
                for y0, y1 in zip(ys[:-1], ys[1:]):
                    box = [x0, x1, y0, y1]
                    for component in (0, 1):
                        index = row(kind="bulk_balance", material=cell["material"], cell=cell_id,
                                    bounds=box, component=component, degree=0)
                        for face in range(4):
                            axis, sign = face//2, (-1 if face % 2 == 0 else 1)
                            lo, hi = (y0, y1) if axis == 0 else (x0, x1)
                            ids = line(cell_id, axis, box[face], lo, hi)
                            field = ((2, 4) if axis == 0 else (4, 3))[component]
                            term(index, cell_id, ids, field, sign*(hi-lo)*w)

        for edge_id, edge in enumerate(model.edges):
            a, b, axis = [edge[k] for k in ("a", "b", "axis")]
            p0, p1 = np.asarray(edge["endpoints"])
            lo, hi, position = p0[1-axis], p1[1-axis], p0[axis]
            cuts = distinct_cuts([lo, hi]+[
                center[1-axis]+k*radius for center in centers
                if abs(position-center[axis]) <= radius+1e-12 for k in (-1, 0, 1)
                if lo < center[1-axis]+k*radius < hi])
            stiffness = model.k.detach().numpy()[::1 if axis == 0 else -1]
            for start, stop in zip(cuts[:-1], cuts[1:]):
                ids_a, ids_b = [line(cell, axis, position, start, stop) for cell in (a, b)]
                for order in range(degree+1):
                    weights = (stop-start)*w*legendre[:, order]
                    for component in (0, 1):
                        field = ((2, 4) if axis == 0 else (4, 3))[component]
                        if edge["interface"]:
                            for cell, ids in ((a, ids_a), (b, ids_b)):
                                index = row(kind="interface_traction", material=model.cells[cell]["material"],
                                            edge=edge_id, side_cell=cell, interval=[start, stop],
                                            component=component, degree=order)
                                term(index, cell, ids, field, weights)
                                term(index, a, ids_a, component, stiffness[component]*weights)
                                term(index, b, ids_b, component, -stiffness[component]*weights)
                        else:
                            index = row(kind="shared_traction", material=model.cells[a]["material"],
                                        edge=edge_id, interval=[start, stop], component=component, degree=order)
                            term(index, a, ids_a, field, weights)
                            term(index, b, ids_b, field, -weights)

        for edge in model.geometry["outer_boundaries"]:
            cell, face = edge["cell"], edge["face"]
            box = model.cells[cell]["bounds"]
            axis = face//2
            lo, hi = box[2:] if axis == 0 else box[:2]
            ids = line(cell, axis, box[face], lo, hi)
            for order in range(degree+1):
                for component in ((0, 1) if axis == 0 else (0,)):
                    field = ((2, 4) if axis == 0 else (4, 3))[component]
                    index = row(kind="natural_traction", material=model.cells[cell]["material"],
                                cell=cell, face=face, component=component, degree=order)
                    term(index, cell, ids, field, (hi-lo)*w*legendre[:, order])

        size = max(point_count)
        padded = []
        for cell, rows in zip(model.cells, points):
            x0, x1, y0, y1 = cell["bounds"]
            xy = np.concatenate(rows)
            padded.append(np.concatenate([xy, np.tile([(x0+x1)/2, (y0+y1)/2], (size-len(xy), 1))]))
        self.points = torch.tensor(np.array(padded), requires_grad=True)
        columns = (np.array(cell_ids)*size+np.array(sample_ids))*5+np.array(field_ids)
        self.operator = sparse.csr_matrix((factors, (row_ids, columns)),
                                          shape=(len(self.rows), len(model.cells)*size*5))

    def __call__(self, displacement):
        uv = displacement(self.points)
        stress = self.model.stress(strain_of(uv, self.points))
        fields = torch.cat([uv, stress], -1).detach().numpy().reshape(-1)
        return torch.from_numpy(self.operator@fields)

    def readout(self):
        residual = self(self.model.displacement).numpy()
        scale = self.model.case.soil_E_MPa*self.model.case.applied_shortening_mm
        groups = {}
        for row, value in zip(self.rows, residual):
            name = row["kind"]+"/"+row["material"]
            groups[name] = max(groups.get(name, 0.), abs(float(value))*scale)
        worst = np.argsort(np.abs(residual))[-8:][::-1]
        return {"rows": len(self.rows), "edge_nodes_per_interval": self.edge_nodes,
                "degree": self.degree, "counts": dict(Counter(r["kind"] for r in self.rows)),
                "maximum_absolute_force_moments_N_per_mm": groups,
                "largest_rows": [{**self.rows[i], "residual_N_per_mm": float(residual[i])*scale} for i in worst]}


def solve_constrained(matrix, base, constraints, constraint_base, rcond):
    """Minimize ||A x+b|| subject to C x+d=0 via an SVD nullspace.

    A is overwritten with its nullspace projection in row blocks to avoid a
    second tall dense matrix. Redundant equations are allowed; incompatible
    equations raise instead of being silently replaced by a penalty fit.
    """
    norms = np.linalg.norm(constraints, axis=1)
    nonzero = norms > 0.
    if np.any(constraint_base[~nonzero] != 0.):
        raise ValueError("A constant equation constraint is nonzero.")
    c = constraints[nonzero]/norms[nonzero, None]
    target = -constraint_base[nonzero]/norms[nonzero]
    u, singular, vt = linalg.svd(c, full_matrices=True)
    constraint_rcond = np.finfo(c.dtype).eps*max(c.shape)
    rank = int((singular > constraint_rcond*singular[0]).sum())
    particular = vt[:rank].T@((u[:, :rank].T@target)/singular[:rank])
    feasibility = np.linalg.norm(c@particular-target)
    if feasibility > 100*constraint_rcond*(1+np.linalg.norm(target)):
        raise ValueError(f"Incompatible retained equation constraints: residual {feasibility:.6g}")
    nullspace = vt[rank:].T
    shifted = base+matrix@particular
    free = nullspace.shape[1]
    if free == 0:
        solution, residual = particular, shifted
        reduced_rank, condition, stationarity = 0, None, 0.
    else:
        for first in range(0, len(base), 2048):
            block = matrix[first:first+2048]@nullspace
            matrix[first:first+2048, :free] = block
        reduced = matrix[:, :free]
        coefficients, _, reduced_rank, singular = linalg.lstsq(
            reduced, -shifted, cond=rcond, lapack_driver="gelsd")
        solution = particular+nullspace@coefficients
        residual = shifted+reduced@coefficients
        condition = float(singular[0]/singular[reduced_rank-1]) if reduced_rank else None
        stationarity = float(np.abs(reduced.T@residual).max())
    return solution, residual, {
        "constraint_rows": len(constraint_base), "constraint_rank": rank,
        "constraint_rcond": float(constraint_rcond),
        "free_columns": free, "rank": int(reduced_rank), "retained_condition": condition,
        "constraint_feasibility_scaled_l2": float(feasibility),
        "constraint_residual_max": float(np.abs(constraints@solution+constraint_base).max()),
        "nullspace_stationarity_max": stationarity}
