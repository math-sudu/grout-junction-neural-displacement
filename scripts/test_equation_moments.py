"""Verify physical moment signs and equality-constrained coefficient fitting."""
import unittest

from equation_moments import EquationMoments, solve_constrained

import numpy as np
from scipy import linalg
import torch

from full_domain_pinn import FullDomainCase, FullDomainNet, full_domain_quadrature
from partitioned_energy_pinn import fit_energy_readout
from traction_control import EquilibriumControl


class EquationMomentTests(unittest.TestCase):
    def test_polynomial_volume_balance_and_shared_traction(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        moments = EquationMoments(model, edge_nodes=6)
        def field(xy):
            return torch.stack([xy[..., 0]**2+2*xy[..., 1],
                                3*xy[..., 1]**2+xy[..., 0]], -1)
        actual = moments(field).numpy()
        nu = model.case.poisson
        factor = (1-nu)/((1+nu)*(1-2*nu))
        for row, value in zip(moments.rows, actual):
            if row["kind"] == "bulk_balance":
                x0, x1, y0, y1 = row["bounds"]
                expected = (2, 6)[row["component"]]*factor*float(model.E[row["cell"], 0, 0])*(x1-x0)*(y1-y0)
                self.assertAlmostEqual(value, expected, places=11)
            elif row["kind"] == "shared_traction":
                self.assertAlmostEqual(value, 0., places=11)

    def test_spring_direction_and_both_sides(self):
        model = FullDomainNet(FullDomainCase(tangent_K_MPa_per_mm=1.), width=1, edge_width=1, corners=False)
        moments = EquationMoments(model, edge_nodes=6)
        shifts = torch.arange(2*len(model.cells)).reshape(-1, 1, 2)/17.
        actual = moments(lambda xy: 0*xy.square()+shifts).numpy()
        for row, value in zip(moments.rows, actual):
            expected = 0.
            if row["kind"] == "interface_traction" and row["degree"] == 0:
                edge = model.edges[row["edge"]]
                stiffness = model.k if edge["axis"] == 0 else model.k.flip(0)
                jump = shifts[edge["b"], 0]-shifts[edge["a"], 0]
                lo, hi = row["interval"]
                expected = -float((stiffness*jump)[row["component"]])*(hi-lo)
            self.assertAlmostEqual(value, expected, places=12)

    def test_nullspace_solution_with_redundant_equations(self):
        rng = np.random.default_rng(29)
        a = rng.normal(size=(17, 7))
        b = rng.normal(size=17)
        c = rng.normal(size=(3, 7))
        d = rng.normal(size=3)
        kkt = np.block([[a.T@a, c.T], [c, np.zeros((3, 3))]])
        reference = linalg.solve(kkt, np.concatenate([-a.T@b, -d]))[:7]
        duplicate_c, duplicate_d = np.vstack([c, 2*c[0]]), np.r_[d, 2*d[0]]
        solution, residual, details = solve_constrained(a.copy(), b, duplicate_c, duplicate_d, 1e-12)
        np.testing.assert_allclose(solution, reference, atol=1e-12)
        np.testing.assert_allclose(residual, a@solution+b, atol=1e-12)
        self.assertEqual(details["constraint_rank"], 3)
        self.assertLess(details["constraint_residual_max"], 1e-12)
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            solve_constrained(a.copy(), b, duplicate_c, np.r_[d, 2*d[0]+1], 1e-12)

    def test_complete_field_matches_constrained_affine_solve(self):
        model = FullDomainNet(width=2, edge_width=2, corners=False)
        quad = list(full_domain_quadrature(model, 5, 8))
        quad[0].requires_grad_(True)
        moments = EquationMoments(model, edge_nodes=8, degree=0)
        # A finite physically defined subset exercises nonhomogeneous loading,
        # trace and interior columns without overconstraining this small net.
        selected = [i for i, row in enumerate(moments.rows) if row["kind"] == "bulk_balance"][:12]
        fit = fit_energy_readout(model, quad, 1e-9, equation_constraints=lambda u: moments(u)[selected])
        self.assertLess(fit["matrix_field_residual_difference"], 1e-7)
        self.assertLess(fit["matrix_field_constraint_difference"], 1e-7)
        self.assertLess(fit["constraint_field_residual_max"], 1e-7)
        self.assertNotIn("load_derivative_reaction_full", fit)
        values = model.displacement(quad[2]).reshape(len(model.cells), 4, 8, 2).detach().numpy()
        for edge in model.edges:
            if not edge["interface"]:
                np.testing.assert_allclose(values[edge["a"], edge["fa"]], values[edge["b"], edge["fb"]], atol=1e-11)
        for edge in model.geometry["outer_boundaries"]:
            if edge["uy"] is not None:
                np.testing.assert_allclose(values[edge["cell"], edge["face"], :, 1], edge["uy"], atol=1e-12)

    def test_bulk_control_keeps_moments_and_reconstructs_joint_objective(self):
        model = FullDomainNet(width=2, edge_width=2, corners=False)
        quad = list(full_domain_quadrature(model, 5, 8))
        quad[0].requires_grad_(True)
        moments = EquationMoments(model, edge_nodes=8, degree=0)
        selected = [i for i, row in enumerate(moments.rows) if row["kind"] == "bulk_balance"][:12]
        constraints = lambda u: moments(u)[selected]
        control = EquilibriumControl(model, 5, 8, equilibrium_metric="reference")
        fit_energy_readout(model, quad, 1e-9, equation_constraints=constraints)
        bulk_before = float(.5*control.bulk_residual(model.displacement).square().sum())
        fit = fit_energy_readout(model, quad, 1e-9, control.bulk_residual, constraints)
        bulk_after = float(.5*control.bulk_residual(model.displacement).square().sum())
        self.assertLess(bulk_after, bulk_before)
        self.assertLess(fit["objective_after"], fit["objective_before"])
        self.assertLess(fit["matrix_field_residual_difference"], 1e-7)
        self.assertLess(fit["matrix_field_constraint_difference"], 1e-7)
        self.assertLess(fit["constraint_field_residual_max"], 1e-7)
        self.assertAlmostEqual(fit["equation_residual_penalty_after"], bulk_after, places=10)
        self.assertAlmostEqual(fit["objective_after"], fit["energy_after"]+bulk_after, places=10)
        self.assertNotIn("load_derivative_reaction_full", fit)

    def test_full_control_keeps_bulk_and_face_moments(self):
        model = FullDomainNet(width=2, edge_width=2, corners=False)
        quad = list(full_domain_quadrature(model, 5, 8))
        quad[0].requires_grad_(True)
        moments = EquationMoments(model, edge_nodes=8, degree=0)
        selected = []
        for kind in ("bulk_balance", "interface_traction", "shared_traction", "natural_traction"):
            selected.extend([i for i, row in enumerate(moments.rows) if row["kind"] == kind][:2])
        constraints = lambda u: moments(u)[selected]
        control = EquilibriumControl(model, 5, 8, interface_metric="spring",
                                     equilibrium_metric="reference")
        fit_energy_readout(model, quad, 1e-9, equation_constraints=constraints)
        fit = fit_energy_readout(model, quad, 1e-9, control, constraints)
        terms = control.readout()["penalty_terms"]
        self.assertTrue(all(value > 0 for value in terms.values()))
        self.assertLess(fit["objective_after"], fit["objective_before"])
        self.assertLess(fit["matrix_field_residual_difference"], 1e-7)
        self.assertLess(fit["matrix_field_constraint_difference"], 1e-7)
        self.assertLess(fit["constraint_field_residual_max"], 1e-7)
        self.assertAlmostEqual(fit["equation_residual_penalty_after"], sum(terms.values()), places=10)
        self.assertAlmostEqual(fit["objective_after"], fit["energy_after"]+sum(terms.values()), places=10)
        self.assertNotIn("load_derivative_reaction_full", fit)


if __name__ == "__main__":
    unittest.main()
