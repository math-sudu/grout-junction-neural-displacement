"""Check C1 admissibility, elastic derivatives, and interior correction fits."""
import unittest
from types import SimpleNamespace

from diagnose_moment_interior import (MomentInteriorField, attach_interior, fit_interior,
                                      local_operator, objective_readout)
from equation_moments import EquationMoments
from full_domain_pinn import FullDomainNet, full_domain_quadrature
from local_virtual_work import strain_of, strain_residual
from traction_control import EquilibriumControl, stress_divergence

import numpy as np
import torch


class MomentInteriorTests(unittest.TestCase):
    def test_zero_trace_gradient_and_all_moments(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        moments = EquationMoments(model, edge_nodes=8)
        rectangles = [{key: row[key] for key in ("cell", "material", "bounds")}
                      for row in moments.rows if row["kind"] == "bulk_balance" and row["component"] == 0]
        field = MomentInteriorField(rectangles, 3)
        field.coefficients = torch.tensor(np.random.default_rng(2).normal(size=field.coefficients.shape))
        points = []
        for row in rectangles:
            x0, x1, y0, y1 = row["bounds"]
            points.extend([[x0, (y0+y1)/2], [x1, (y0+y1)/2], [(x0+x1)/2, y0], [(x0+x1)/2, y1]])
        # Check each rectangle in isolation: another rectangle can have support
        # at these same coordinates in a different material/cell.
        for row, coefficients in zip(rectangles, field.coefficients):
            isolated = MomentInteriorField([row], 3, coefficients[None].tolist())
            xy = torch.tensor(np.tile(points, (len(model.cells), 1, 1)), requires_grad=True)
            own = 4*rectangles.index(row)
            value = isolated(xy)
            np.testing.assert_allclose(value[row["cell"], own:own+4].detach(), 0, atol=1e-26)
            for component in (0, 1):
                gradient = torch.autograd.grad(value[..., component].sum(), xy, retain_graph=True)[0]
                np.testing.assert_allclose(gradient[row["cell"], own:own+4], 0, atol=1e-24)
        np.testing.assert_allclose(moments(field), 0, atol=1e-11)

    def test_local_columns_match_full_constitutive_operators(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        cell = 7
        bounds = model.cells[cell]["bounds"]
        x0, x1, y0, y1 = bounds
        local = np.random.default_rng(5).uniform(.1, .9, (17, 2))
        points = local*np.array([x1-x0, y1-y0])+[x0, y0]
        weights = np.linspace(.01, .03, len(points))
        coordinates = np.tile(points, (len(model.cells), 1, 1))
        quad_weights = torch.zeros(len(model.cells), len(points))
        quad_weights[cell] = torch.tensor(weights)
        field = MomentInteriorField([{"cell": cell, "material": "grout", "bounds": bounds}], 2)
        energy = local_operator(points, weights, bounds, 2, float(model.E[cell, 0, 0]), model.case.poisson)
        bulk = local_operator(points, weights, bounds, 2, float(model.E[cell, 0, 0]), model.case.poisson, .1)
        for j in range(field.coefficients.numel()):
            field.coefficients.zero_()
            field.coefficients.reshape(-1)[j] = 1
            xy = torch.tensor(coordinates, requires_grad=True)
            strain = strain_of(field(xy), xy)
            actual_energy = strain_residual(model, strain, quad_weights).reshape(len(model.cells), -1, 3)[cell]
            np.testing.assert_allclose(energy[:, j], actual_energy.detach().numpy().reshape(-1), atol=1e-10)
            divergence = stress_divergence(model.stress(strain), xy)
            actual_bulk = .1*weights[:, None]**.5*divergence[cell].detach().numpy()
            np.testing.assert_allclose(bulk[:, j], actual_bulk.reshape(-1), atol=1e-9)

    def test_fit_removes_known_interior_perturbation(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        cell = 2
        rectangle = {"cell": cell, **model.cells[cell]}
        known = MomentInteriorField([rectangle], 2)
        known.coefficients = torch.tensor(np.random.default_rng(9).normal(size=known.coefficients.shape)/100)
        model.displacement = lambda xy, *args: torch.stack([.2*xy[..., 0], -xy[..., 1]], -1)+known(xy)
        quad = list(full_domain_quadrature(model, 12, 12))
        quad[0].requires_grad_(True)
        control = EquilibriumControl(model, 12, 12, interface_metric="spring", equilibrium_metric="reference")
        moments = SimpleNamespace(rows=[{**rectangle, "kind": "bulk_balance", "component": 0}])
        before = objective_readout(model, quad, control)
        field, rows = fit_interior(model, quad, control, moments, 2, 1e-12)
        np.testing.assert_allclose(field.coefficients, -known.coefficients, atol=1e-13)
        attach_interior(model, field)
        after = objective_readout(model, quad, control)
        self.assertAlmostEqual(before["objective"]-after["objective"], rows[0]["joint_objective_reduction"], places=10)
        self.assertLess(after["control"]["penalty_terms"]["bulk_equilibrium"], 1e-25)


if __name__ == "__main__":
    unittest.main()
