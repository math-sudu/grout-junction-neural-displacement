"""Verify interior row placement and condensed versus direct coupled fitting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from couple_moment_interior import InteriorElimination, load_parent
from diagnose_moment_interior import attach_interior, load_probe_field
from equation_moments import EquationMoments
from full_domain_pinn import FullDomainNet, full_domain_quadrature
from local_virtual_work import displacement_residual
from partitioned_energy_pinn import fit_energy_readout
from traction_control import EquilibriumControl

import numpy as np
from scipy import linalg
import torch


class CoupledInteriorTests(unittest.TestCase):
    def test_stiffness_change_preserves_trial_field_and_checkpoint_case(self):
        original = FullDomainNet(width=2, edge_width=2, corners=False)
        xy = full_domain_quadrature(original, 4, 4)[0]
        expected = original.displacement(xy).detach()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"parent.pt"
            original.save(checkpoint)
            changed = load_parent(checkpoint, 1.)
            self.assertEqual(changed.options, original.options)
            self.assertEqual(changed.case.normal_K_MPa_per_mm, 10.)
            self.assertEqual(changed.case.tangent_K_MPa_per_mm, 1.)
            np.testing.assert_array_equal(changed.displacement(xy).detach(), expected)
            for name, value in original.state_dict().items():
                if name != "k":
                    torch.testing.assert_close(changed.state_dict()[name], value, rtol=0, atol=0)
            np.testing.assert_allclose(changed.k, original.k.numpy()*[1., .1])
            updated = Path(directory)/"updated.pt"
            changed.save(updated)
            replay = load_parent(updated)
            self.assertEqual(replay.case, changed.case)
            np.testing.assert_array_equal(replay.k, changed.k)
            np.testing.assert_array_equal(replay.displacement(xy).detach(), expected)
            for invalid in (0., -1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    load_parent(checkpoint, invalid)

    def problem(self):
        model = FullDomainNet(width=2, edge_width=2, corners=False,
                              cell_interior={"radius": .1, "width": 1, "seed": 17})
        quad = list(full_domain_quadrature(model, 8, 8, 4))
        quad[0].requires_grad_(True)
        control = EquilibriumControl(model, 8, 8, 4, interface_metric="spring", equilibrium_metric="reference")
        moments = EquationMoments(model, edge_nodes=8, degree=0)
        rectangles = [next(row for row in moments.rows
                           if row["kind"] == "bulk_balance" and row["component"] == 0 and row["cell"] == cell)
                      for cell in (1, 7)]
        elimination = InteriorElimination(model, quad, control, SimpleNamespace(rows=rectangles), 2, 1e-12)
        return model, quad, control, elimination

    def test_compact_row_projection_and_direct_constrained_solution(self):
        model, quad, control, elimination = self.problem()
        self.assertTrue((quad[1] == 0).any())
        def residual(displacement):
            return torch.cat([displacement_residual(model, displacement, quad), control(displacement)]).detach().numpy()
        value = residual(elimination.field)
        keep = np.ones(len(value), dtype=bool)
        keep[:3*quad[1].numel()] = quad[1].numpy().reshape(-1).repeat(3) > 0
        compact = np.cumsum(keep)-1
        block_matrix = np.zeros((keep.sum(), elimination.field.coefficients.numel()))
        cursor = 0
        for block in elimination.blocks:
            columns = block["inverse"].shape[0]
            block_matrix[compact[block["rows"]], cursor:cursor+columns] = block["q"]@linalg.inv(block["inverse"])
            cursor += columns
        rng = np.random.default_rng(41)
        interior_coefficients = rng.normal(size=block_matrix.shape[1])
        elimination.field.coefficients = torch.tensor(interior_coefficients.reshape(elimination.field.coefficients.shape))
        np.testing.assert_allclose(residual(elimination.field)[keep], block_matrix@interior_coefficients,
                                   rtol=1e-11, atol=1e-11)
        a, b = rng.normal(size=(keep.sum(), 9)), rng.normal(size=keep.sum())
        constraints, target = rng.normal(size=(2, 9)), rng.normal(size=2)
        # Independent direct KKT solve includes every interior column.
        joint = np.column_stack([a, block_matrix])
        c = np.column_stack([constraints, np.zeros((2, block_matrix.shape[1]))])
        hessian = joint.T@joint
        kkt = np.block([[hessian, c.T], [c, np.zeros((2, 2))]])
        expected = linalg.solve(kkt, np.r_[-joint.T@b, target])[:-2]
        projected_a, projected_b = a.copy(), b.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            elimination.condense(projected_a, projected_b, keep)
        from equation_moments import solve_constrained
        coefficients, projected_residual, _ = solve_constrained(
            projected_a, projected_b, constraints, -target, 1e-12)
        field = elimination.recover(coefficients)
        actual = np.r_[coefficients, field.coefficients.numpy().reshape(-1)]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(projected_residual, joint@actual+b, rtol=1e-10, atol=1e-10)

    def test_complete_field_solve_and_replay(self):
        model, quad, control, elimination = self.problem()
        moments = EquationMoments(model, edge_nodes=8, degree=0)
        selected = [i for i, row in enumerate(moments.rows) if row["kind"] == "bulk_balance"][:8]
        with contextlib.redirect_stdout(io.StringIO()):
            fit = fit_energy_readout(model, quad, 1e-10, control,
                                     lambda u: moments(u)[selected], elimination)
        self.assertLess(fit["matrix_field_residual_difference"], 1e-7)
        self.assertLess(fit["constraint_field_residual_max"], 1e-7)
        self.assertEqual(fit["interior_elimination"]["columns"], 24)
        self.assertLess(fit["objective_after"], fit["objective_before"])
        np.testing.assert_allclose(moments(elimination.field), 0, atol=1e-11)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"neural.pt"
            model.save(checkpoint)
            result = Path(directory)/"coupled.json"
            result.write_text(json.dumps({"execution": {"parent_checkpoint_ref": "unused.pt",
                                                         "neural_checkpoint_ref": checkpoint.as_posix()},
                                          "interior_field": elimination.field.export()}), encoding="utf-8")
            attach_interior(model, elimination.field)
            replay = load_probe_field(result)
            expected, actual = model(quad[0]), replay(quad[0])
            np.testing.assert_allclose(actual.detach(), expected.detach(), atol=1e-13)
            total = .5*torch.cat([displacement_residual(model, model.displacement, quad),
                                 control(model.displacement)]).detach().square().sum()
            self.assertAlmostEqual(float(total), fit["objective_after"], places=10)


if __name__ == "__main__":
    unittest.main()
