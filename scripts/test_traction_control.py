"""Check equation signs, scales and affine reconstruction of traction control."""
import unittest

import numpy as np
import torch

from full_domain_pinn import FullDomainCase, FullDomainNet, full_domain_quadrature
from partitioned_energy_pinn import fit_energy_readout
from traction_control import EquilibriumControl, junction_balance_readout, stress_divergence


class TractionControlTests(unittest.TestCase):
    def test_cellwise_interior_columns_match_complete_equations(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False,
                              additional_local={"radius": .2, "width": 2, "seed": 29,
                                                "split_interior": True, "cellwise_interior": True})
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.readout_parameters():
            p.zero_()
        parameter = model.additional_local.interior_coefficients
        control = EquilibriumControl(model, 5, 6, interface_metric="spring")
        for corner, cell in ((0, 6), (0, 11), (1, 12), (1, 1)):
            parameter[corner, cell].normal_()
            whole = control(lambda q: model.displacement(q, homogeneous=True))
            selected = control(model.readout_displacement(parameter))
            torch.testing.assert_close(whole, selected, atol=1e-11, rtol=1e-11)

    def test_restricted_base_columns_match_complete_equations(self):
        model = FullDomainNet(width=2, edge_width=2, corners=False)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.readout_parameters():
            p.zero_()
        control = EquilibriumControl(model, 3, 5, interface_metric="spring")
        for p in model.readout_parameters():
            for ids in ([], [0], [p.shape[0]-1], [0, p.shape[0]//2, p.shape[0]-1]):
                p.zero_()
                p[ids] = torch.randn_like(p[ids])
                whole = control(lambda q: model.displacement(q, homogeneous=True))
                selected = control(model.readout_displacement(p))
                torch.testing.assert_close(whole, selected, atol=1e-12, rtol=1e-11)
            p.zero_()

    def test_spring_metric_uses_complementary_energy_on_both_sides(self):
        model = FullDomainNet(FullDomainCase(normal_K_MPa_per_mm=10., tangent_K_MPa_per_mm=1.),
                              width=2, edge_width=2, corners=False)
        control = EquilibriumControl(model, 3, 5, interface_metric="spring")
        shifts = torch.arange(40).reshape(20, 1, 2)/17.
        blocks = control.blocks(lambda xy: 0*xy**2+shifts)
        actual = .5*sum(value.square().sum() for name, _, value in blocks if name == "interface_traction")
        expected = 0.
        for edge in model.edges:
            if edge["interface"]:
                a, b, axis, fa = [edge[k] for k in ("a", "b", "axis", "fa")]
                k = model.k if axis == 0 else model.k.flip(0)
                jump = shifts[b]-shifts[a]
                # Zero stress leaves one copy of the spring energy per side.
                expected += (control.quad[3][a, fa, :, None]*k*jump.square()).sum()
        torch.testing.assert_close(actual, expected)
        faces = [value for name, _, value in blocks if name == "interface_traction"]
        for a, b in zip(faces[::2], faces[1::2]):
            torch.testing.assert_close(a, b)
        readout = control.readout()
        for name, value in readout["penalty_terms"].items():
            self.assertAlmostEqual(value, sum(row[name] for row in readout["material_penalty_terms"].values()))

    def test_polynomial_bulk_and_face_equations(self):
        model = FullDomainNet(FullDomainCase(), width=2, edge_width=2, corners=False)
        control = EquilibriumControl(model, 4, 6)
        def displacement(xy):
            return torch.stack([xy[..., 0]**2+2*xy[..., 1],
                                3*xy[..., 1]**2+xy[..., 0]], -1)
        terms = control.terms(displacement)
        xy, w, faces, wf = control.quad
        nu, young, length = model.case.poisson, model.E[:, 0, 0], control.length
        lam = young*nu/((1+nu)*(1-2*nu))
        mu = young/(2*(1+nu))
        # For u=(x^2+2y,3y^2+x), div(sigma)=(2(lambda+2mu),6(lambda+2mu)).
        d = torch.stack([2*(lam+2*mu), 6*(lam+2*mu)], -1)
        bulk = (length*(w/young[:, None]).sqrt()[..., None]*d[:, None]).flatten()
        torch.testing.assert_close(terms["bulk_equilibrium"], bulk)
        torch.testing.assert_close(control.bulk_residual(displacement), bulk)
        x, y = faces[..., 0], faces[..., 1]
        sigma = torch.stack([lam[:, None]*(2*x+6*y)+4*mu[:, None]*x,
                             lam[:, None]*(2*x+6*y)+12*mu[:, None]*y,
                             3*mu[:, None].expand_as(x)], -1).reshape(20, 4, 6, 3)
        expected = []
        for edge in model.edges:
            if not edge["interface"]:
                continue
            a, b, axis, fa, fb = [edge[k] for k in ("a", "b", "axis", "fa", "fb")]
            indices = [0, 2] if axis == 0 else [2, 1]
            # This smooth displacement has zero spring jump.
            for cell, face in ((a, fa), (b, fb)):
                expected.append(((length*wf[a, fa]/young[cell]).sqrt()[:, None]
                                 *sigma[cell, face][:, indices]).flatten())
        torch.testing.assert_close(terms["interface_traction"], torch.cat(expected))
        torch.testing.assert_close(terms["shared_traction"], torch.zeros_like(terms["shared_traction"]), atol=1e-12, rtol=0)
        expected = []
        for edge in model.geometry["outer_boundaries"]:
            cell, face = edge["cell"], edge["face"]
            indices = [0, 2] if edge["side"] in ("left", "right") else [2]
            expected.append(((length*wf[cell, face]/young[cell]).sqrt()[:, None]
                             *sigma[cell, face][:, indices]).flatten())
        torch.testing.assert_close(terms["natural_traction"], torch.cat(expected))

    def test_reference_metric_uses_common_stress_scale_for_bulk_and_cuts(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        control = EquilibriumControl(model, 4, 6, interface_metric="spring",
                                     equilibrium_metric="reference")
        xy, weights, faces, wf = control.quad
        young = model.E[:, 0, 0]
        # Divide displacement by E_m: stress and its divergence have the same
        # physical magnitude in soil and grout, despite the modulus contrast.
        def displacement(q):
            return torch.stack([q[..., 0]**2, 3*q[..., 1]**2], -1)/young[:, None, None]
        terms = control.terms(displacement)
        nu, length = model.case.poisson, control.length
        lam = nu/((1+nu)*(1-2*nu))
        mu = 1/(2*(1+nu))
        divergence = torch.tensor([2*(lam+2*mu), 6*(lam+2*mu)])
        expected = (length*weights.sqrt()[..., None]*divergence).flatten()
        torch.testing.assert_close(terms["bulk_equilibrium"], expected)
        torch.testing.assert_close(control.bulk_residual(displacement), expected)

        # Cell-dependent amplitudes give nonzero traction jumps on shared
        # cuts; their weights use E_s on both sides, independent of phase.
        amplitudes = torch.arange(len(model.cells), dtype=xy.dtype)+1
        varied = lambda q: displacement(q)*amplitudes[:, None, None]
        terms = control.terms(varied)
        x, y = faces[..., 0], faces[..., 1]
        stress = torch.stack([lam*(2*x+6*y)+4*mu*x,
                              lam*(2*x+6*y)+12*mu*y, 0*x], -1)
        stress = (stress*amplitudes[:, None, None]).reshape(len(model.cells), 4, 6, 3)
        expected = []
        for edge in model.edges:
            if edge["interface"]:
                continue
            a, b, axis, fa, fb = [edge[k] for k in ("a", "b", "axis", "fa", "fb")]
            components = [0, 2] if axis == 0 else [2, 1]
            jump = stress[a, fa][:, components]-stress[b, fb][:, components]
            expected.append(((length*wf[a, fa]/2).sqrt()[:, None]*jump).flatten())
        torch.testing.assert_close(terms["shared_traction"], torch.cat(expected))
        material = EquilibriumControl(model, 4, 6, interface_metric="spring")
        torch.testing.assert_close(terms["interface_traction"], material.terms(varied)["interface_traction"])

    def test_spring_sign_and_directional_stiffness(self):
        model = FullDomainNet(FullDomainCase(normal_K_MPa_per_mm=10., tangent_K_MPa_per_mm=1.),
                              width=2, edge_width=2, corners=False)
        control = EquilibriumControl(model, 3, 5)
        shifts = torch.randn(20, 1, 2)
        # Independent rigid translations isolate both sides of the spring residual.
        terms = control.terms(lambda xy: 0*xy**2+shifts)
        expected = []
        for edge in model.edges:
            if edge["interface"]:
                a, b, axis, fa = [edge[k] for k in ("a", "b", "axis", "fa")]
                spring = (model.k if axis == 0 else model.k.flip(0))*(shifts[b]-shifts[a])
                for cell in (a, b):
                    expected.append((-(control.length*control.quad[3][a, fa]/model.E[cell, 0, 0]).sqrt()[:, None]*spring).flatten())
        torch.testing.assert_close(terms["interface_traction"], torch.cat(expected))
        torch.testing.assert_close(terms["bulk_equilibrium"], torch.zeros_like(terms["bulk_equilibrium"]))
        xy = control.quad[0]
        torch.testing.assert_close(stress_divergence(torch.ones(20, 9, 3), xy), torch.zeros_like(xy))

    def test_augmented_solve_preserves_affine_constraints_and_reconstruction(self):
        model = FullDomainNet(width=1, edge_width=1, corners=True,
                              local={"radius": .1, "width": 1, "seed": 17})
        quad = list(full_domain_quadrature(model, 4, 8))
        quad[0].requires_grad_(True)
        control = EquilibriumControl(model, 4, 8, interface_metric="spring",
                                     equilibrium_metric="reference")
        fit = fit_energy_readout(model, quad, 1e-9, control)
        self.assertLess(fit["objective_after"], fit["objective_before"])
        self.assertLess(fit["matrix_field_residual_difference"], 1e-7)
        self.assertAlmostEqual(fit["objective_after"], fit["energy_after"]+fit["equation_residual_penalty_after"], places=10)
        self.assertNotIn("load_derivative_reaction_full", fit)
        values = model.displacement(quad[2]).reshape(20, 4, 8, 2).detach().numpy()
        for edge in model.edges:
            if not edge["interface"]:
                np.testing.assert_allclose(values[edge["a"], edge["fa"]], values[edge["b"], edge["fb"]], atol=1e-11)
        for edge in model.geometry["outer_boundaries"]:
            if edge["uy"] is not None:
                np.testing.assert_allclose(values[edge["cell"], edge["face"], :, 1], edge["uy"], atol=1e-12)

    def test_junction_control_volume_normals_and_internal_pairing(self):
        model = FullDomainNet(width=1, edge_width=1, corners=False)
        class PiecewiseStress:
            def __init__(self, source):
                self.case, self.cells, self.geometry = source.case, source.cells, source.geometry
                self.edges = source.edges
                self.k, self.vein_width = source.k, source.vein_width
                self.values = torch.randn(20, 1, 3)

            def __call__(self, xy):
                return torch.cat([torch.zeros_like(xy), self.values.expand(-1, xy.shape[1], -1)], -1)

        field = PiecewiseStress(model)
        for row in junction_balance_readout(field, 4):
            self.assertAlmostEqual(row["grout_area_over_H2"], .0075)
            self.assertEqual(sum(f["role"] == "physical_interface" for f in row["faces"]), 2)
            # Constant stress has zero divergence inside each cell: external
            # balance plus the internal traction-jump sum cancels exactly.
            np.testing.assert_allclose(np.array(row["constitutive_boundary_net_xy"])
                                       +row["internal_cut_net_xy"], 0., atol=1e-13)
        field.values[:] = torch.tensor([2., -3., .7])
        for row in junction_balance_readout(field, 4):
            np.testing.assert_allclose(row["constitutive_boundary_net_xy"], 0., atol=1e-13)
            np.testing.assert_allclose(row["internal_cut_net_xy"], 0., atol=1e-13)

    def test_complete_enrichment_derivatives_and_column_selection(self):
        settings = {"radius": .05, "width": 1, "seed": 43}
        model = FullDomainNet(width=1, edge_width=1, corners=True,
                              local={"radius": .1, "width": 1, "seed": 17},
                              additional_local={"radius": .2, "width": 1, "seed": 29,
                                                "split_interior": True, "cellwise_interior": True},
                              interface_local=settings, cell_interior=settings, shared_local=settings,
                              junction_patches=[{"radius": .02, "width": 1, "seed": 59}])
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.readout_parameters():
            p.copy_(torch.randn_like(p)*.01)
        xy = full_domain_quadrature(model, 3, 5)[0].requires_grad_(True)
        stress = model(xy)[..., 2:]
        tracked = [torch.autograd.grad(stress[..., j].sum(), xy, create_graph=True)[0]
                   for j in range(3)]
        tracked_divergence = torch.stack([tracked[0][..., 0]+tracked[2][..., 1],
                                         tracked[2][..., 0]+tracked[1][..., 1]], -1)
        divergence = stress_divergence(stress, xy)
        torch.testing.assert_close(divergence, tracked_divergence, atol=0., rtol=0.)
        h = 1e-6
        dx, dy = torch.tensor([h, 0.]), torch.tensor([0., h])
        dsx = (model(xy+dx)[..., 2:]-model(xy-dx)[..., 2:])/(2*h)
        dsy = (model(xy+dy)[..., 2:]-model(xy-dy)[..., 2:])/(2*h)
        finite = torch.stack([dsx[..., 0]+dsy[..., 2], dsx[..., 2]+dsy[..., 1]], -1)
        torch.testing.assert_close(divergence, finite, atol=2e-3, rtol=2e-5)
        control = EquilibriumControl(model, 3, 5)
        for p in model.readout_parameters():
            p.zero_()
        for p in model.readout_parameters():
            p.reshape(-1)[p.numel()//2] = 1.
            whole = control(lambda q: model.displacement(q, homogeneous=True))
            selected = control(model.readout_displacement(p))
            torch.testing.assert_close(whole, selected, atol=1e-11, rtol=1e-10)
            p.zero_()


if __name__ == "__main__":
    unittest.main()
