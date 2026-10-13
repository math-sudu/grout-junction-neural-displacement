"""Energy, virtual work and shared-trace checks for the neural primal solve."""
from copy import deepcopy
import unittest

from partitioned_energy_pinn import EnergyNet, energy_residual, fit_energy_readout
from partitioned_mixed_pinn import Case, PatchNet, evaluate, quadrature
from junction_enrichment import JunctionModes, wedge_displacement
from local_neural_enrichment import LocalNeuralModes, adapt_local_features
from local_virtual_work import displacement_residual, local_probe
import numpy as np
import torch


class EnergyTests(unittest.TestCase):
    def test_additive_support_keeps_the_existing_local_field(self):
        model = EnergyNet(Case(geometry="cross"), width=2,
                          local_neural={"radius": .075, "width": 2})
        with torch.no_grad():
            model.local.coefficients.normal_(std=.02)
        quad = list(quadrature(model.cells, 6, 9))
        quad[0].requires_grad_(True)
        before = model.displacement(quad[0]).detach()
        model.additional_local = LocalNeuralModes(model.case, model.cells, model.vein_width/2,
                                                   .15, width=2, seed=29)
        torch.testing.assert_close(model.displacement(quad[0]), before)
        info = fit_energy_readout(model, quad, 1e-8)
        self.assertLess(info["matrix_field_residual_difference"], 1e-7)
        self.assertLess(info["energy_after"], info["energy_before"])
        self.assertGreater(float(model.additional_local.coefficients.detach().abs().max()), 1e-8)

    def model(self):
        model = EnergyNet(Case(geometry="cross", normal_K_MPa_per_mm=7.,
                               tangent_K_MPa_per_mm=3.), width=4)
        with torch.no_grad():
            for p in model.readout_parameters():
                p.add_(torch.randn_like(p)*.03)
        return model

    def test_mixed_displacement_import_and_constitutive_stress(self):
        for corners, elastic in [(False, False), (True, False), (False, True)]:
            mixed = PatchNet(Case(geometry="cross"), width=4, kinematics="shared_trace",
                             displacement_scaling="compliance",
                             corner_enrichment=corners, elastic_corners=elastic)
            with torch.no_grad():
                for p in mixed.parameters():
                    p.add_(torch.randn_like(p)*.01)
            model = EnergyNet(mixed.case, 4, displacement_scaling="compliance",
                              corner_enrichment=corners, elastic_corners=elastic)
            state = deepcopy(mixed.state_dict())
            for key in ("weights.2", "biases.2"):
                state[key] = state[key][..., :2].clone()
            model.load_state_dict(state)
            xy, _, faces, _ = quadrature(model.cells, 6, 7)
            xy.requires_grad_(True)
            torch.testing.assert_close(model.displacement(xy), mixed(xy)[..., :2])
            f, eps, _, _ = model.fields(xy)
            nu = model.case.poisson
            recovered = (1+nu)/model.E*torch.stack([
                (1-nu)*f[..., 2]-nu*f[..., 3],
                (1-nu)*f[..., 3]-nu*f[..., 2], 2*f[..., 4]], -1)
            torch.testing.assert_close(eps, recovered)
            with torch.no_grad():
                pe = model(faces).reshape(len(model.cells), 4, -1, 5)
            for i, cell in enumerate(model.cells):
                b = cell["bounds"]
                if b[0] == 0.:
                    torch.testing.assert_close(pe[i, 0, :, 0], torch.zeros(7), atol=1e-13, rtol=0.)
                if b[2] == 0.:
                    torch.testing.assert_close(pe[i, 2, :, 1], torch.zeros(7), atol=1e-13, rtol=0.)
                if b[3] == .5:
                    torch.testing.assert_close(pe[i, 3, :, 1], torch.full((7,), -.5), atol=1e-13, rtol=0.)
            for e in model.edges:
                if not e["interface"]:
                    torch.testing.assert_close(pe[e["a"], e["fa"], :, :2],
                                               pe[e["b"], e["fb"], :, :2], atol=1e-13, rtol=0.)

    def test_energy_and_virtual_work_with_directional_springs(self):
        model = self.model()
        quad = list(quadrature(model.cells, 8, 10))
        xy, w, faces, wf = quad
        xy.requires_grad_(True)
        _, eps = model.displacement_strain(xy)
        stress = model.stress(eps)
        energy = .5*(w*(stress*eps).sum(-1)).sum()
        perturbation = deepcopy(model)
        with torch.no_grad():
            for p in perturbation.readout_parameters():
                p.normal_(std=.02)
        _, deps = perturbation.displacement_strain(xy, True)
        virtual_work = (w*(stress*deps).sum(-1)).sum()
        uv = model.displacement(faces).reshape(len(model.cells), 4, -1, 2)
        du = perturbation.displacement(faces, True).reshape_as(uv)
        for e in model.edges:
            if not e["interface"]:
                continue
            a, b, ax, fa, fb = [e[k] for k in ("a", "b", "axis", "fa", "fb")]
            k = model.k if ax == 0 else model.k.flip(0)
            jump, djump = uv[b, fb]-uv[a, fa], du[b, fb]-du[a, fa]
            energy = energy+.5*(wf[a, fa]*(k*jump**2).sum(-1)).sum()
            virtual_work = virtual_work+(wf[a, fa]*(k*jump*djump).sum(-1)).sum()
        r = energy_residual(model, quad)
        torch.testing.assert_close(.5*r.square().sum(), energy)
        grad = torch.autograd.grad(.5*r.square().sum(), model.readout_parameters())
        directional = sum((a*b).sum() for a, b in zip(grad, perturbation.readout_parameters()))
        torch.testing.assert_close(directional, virtual_work)

    def test_free_boundary_work_in_integration_by_parts(self):
        response = evaluate(self.model(), 18, 32, 8)[0]
        defect = response["virtual_work_defect_terms_over_boundary_work"]
        self.assertGreater(abs(defect["natural_boundary_work"]), .001)
        self.assertAlmostEqual(1-response["internal_virtual_work_over_boundary_work"],
                               sum(defect.values()), places=9)

    def test_fixed_feature_energy_solve_and_loading_derivative(self):
        model = EnergyNet(Case(), width=3)
        quad = list(quadrature(model.cells, 9, 12))
        quad[0].requires_grad_(True)
        info = fit_energy_readout(model, quad, 1e-8)
        self.assertLess(info["energy_after"], info["energy_before"])
        self.assertLess(info["matrix_field_residual_difference"], 1e-7)
        self.assertLess(info["normalized_column_stationarity_max"], 1e-7)
        self.assertAlmostEqual(info["load_derivative_reaction_quarter"],
                               4*info["energy_after"], places=6)
        with torch.no_grad():
            for p in model.readout_parameters():
                p.add_(torch.randn_like(p)*.01)
        changed = .5*energy_residual(model, quad).square().sum()
        self.assertGreater(float(changed.detach()), info["energy_after"])

    def test_wedge_modes_equilibrium_and_free_leading_traction(self):
        case = Case(geometry="cross")
        model = EnergyNet(case, width=3, junction_radius=.075)
        angles = torch.tensor([np.pi/2, np.pi, 5*np.pi/4, 1.8*np.pi, 2*np.pi])
        points = (.04*torch.stack([angles.cos(), angles.sin()], -1)).requires_grad_(True)
        nu = case.poisson
        lam, mu = nu/((1+nu)*(1-2*nu)), 1/(2*(1+nu))
        for j, exponent in enumerate(model.junction.exponents):
            uv = wedge_displacement(points[:, 0], points[:, 1], exponent, j == 0, nu)
            du, dv = [torch.autograd.grad(uv[:, k].sum(), points, create_graph=True)[0]
                      for k in range(2)]
            dilation = du[:, 0]+dv[:, 1]
            stress = torch.stack([lam*dilation+2*mu*du[:, 0],
                                  lam*dilation+2*mu*dv[:, 1],
                                  mu*(du[:, 1]+dv[:, 0])], -1)
            ds = [torch.autograd.grad(stress[:, k].sum(), points, retain_graph=True)[0]
                  for k in range(3)]
            divergence = torch.stack([ds[0][:, 0]+ds[2][:, 1],
                                      ds[2][:, 0]+ds[1][:, 1]], -1)
            torch.testing.assert_close(divergence, torch.zeros_like(divergence), atol=1e-10, rtol=0.)
            torch.testing.assert_close(stress[0, [0, 2]], torch.zeros(2), atol=1e-10, rtol=0.)
            torch.testing.assert_close(stress[-1, [2, 1]], torch.zeros(2), atol=1e-10, rtol=0.)

    def test_junction_modes_preserve_essential_conditions_and_common_traces(self):
        model = EnergyNet(Case(geometry="cross"), width=3, elastic_corners=True)
        xy, _, faces, _ = quadrature(model.cells, 5, 9)
        before = model.displacement(xy).detach()
        model.junction = JunctionModes(model.case, model.cells, model.vein_width/2, .075)
        torch.testing.assert_close(model.displacement(xy), before)
        with torch.no_grad():
            model.junction.coefficients.copy_(torch.tensor([.8, -.3]))
        correction = model.junction(faces).reshape(len(model.cells), 4, -1, 2)
        for i, cell in enumerate(model.cells):
            if cell["material"] == "soil":
                torch.testing.assert_close(correction[i], torch.zeros_like(correction[i]))
            if cell["bounds"][0] == 0.:
                torch.testing.assert_close(correction[i, 0, :, 0], torch.zeros(9), atol=1e-13, rtol=0.)
            if cell["bounds"][2] == 0.:
                torch.testing.assert_close(correction[i, 2, :, 1], torch.zeros(9), atol=1e-13, rtol=0.)
            if cell["bounds"][3] == .5:
                torch.testing.assert_close(correction[i, 3, :, 1], torch.zeros(9), atol=1e-13, rtol=0.)
        for edge in model.edges:
            if not edge["interface"]:
                torch.testing.assert_close(correction[edge["a"], edge["fa"]],
                                           correction[edge["b"], edge["fb"]])
        quad = list(quadrature(model.cells, 9, 12))
        quad[0].requires_grad_(True)
        info = fit_energy_readout(model, quad, 1e-8)
        self.assertLess(info["matrix_field_residual_difference"], 1e-7)
        self.assertAlmostEqual(info["load_derivative_reaction_quarter"],
                               4*info["energy_after"], places=6)

    def test_local_probe_decomposition_and_directional_virtual_work(self):
        model = self.model()
        quad = list(quadrature(model.cells, 9, 12))
        quad[0].requires_grad_(True)
        probes = {part: (lambda xy, part=part: local_probe(
            model, xy, model.vein_width, "grout", 1, "y", part))
            for part in ("full", "bubble", "trace")}
        vectors = {part: displacement_residual(model, probe, quad)
                   for part, probe in probes.items()}
        torch.testing.assert_close(vectors["full"], vectors["bubble"]+vectors["trace"])
        bulk_size = 3*quad[0].shape[0]*quad[0].shape[1]
        torch.testing.assert_close(vectors["bubble"][bulk_size:],
                                   torch.zeros_like(vectors["bubble"][bulk_size:]))
        coefficient = torch.tensor(0., requires_grad=True)
        perturbed = lambda xy: model.displacement(xy)+coefficient*probes["full"](xy)
        energy = .5*displacement_residual(model, perturbed, quad).square().sum()
        work, = torch.autograd.grad(energy, coefficient)
        torch.testing.assert_close(work, energy_residual(model, quad)@vectors["full"])

    def test_local_neural_field_constraints_and_energy_reconstruction(self):
        model = EnergyNet(Case(geometry="cross"), width=3, elastic_corners=True,
                          junction_radius=.15)
        xy, _, faces, _ = quadrature(model.cells, 5, 9)
        before = model.displacement(xy).detach()
        model.local = LocalNeuralModes(model.case, model.cells, model.vein_width/2, .075, width=4)
        torch.testing.assert_close(model.displacement(xy), before)
        with torch.no_grad():
            model.local.coefficients.normal_(std=.1)
        correction = model.local(faces).reshape(len(model.cells), 4, -1, 2)
        for i, cell in enumerate(model.cells):
            if cell["bounds"][0] == 0.:
                torch.testing.assert_close(correction[i, 0, :, 0], torch.zeros(9), atol=1e-13, rtol=0.)
            if cell["bounds"][2] == 0.:
                torch.testing.assert_close(correction[i, 2, :, 1], torch.zeros(9), atol=1e-13, rtol=0.)
            if cell["bounds"][3] == .5:
                torch.testing.assert_close(correction[i, 3, :, 1], torch.zeros(9), atol=1e-13, rtol=0.)
        for edge in model.edges:
            if not edge["interface"]:
                torch.testing.assert_close(correction[edge["a"], edge["fa"]],
                                           correction[edge["b"], edge["fb"]])
        interface = next(e for e in model.edges if e["interface"] and e["a"] == 3)
        self.assertGreater(float((correction[interface["a"], interface["fa"]]
                                  -correction[interface["b"], interface["fb"]]).abs().max().detach()), 1e-8)
        quad = list(quadrature(model.cells, 9, 12))
        quad[0].requires_grad_(True)
        info = fit_energy_readout(model, quad, 1e-8)
        self.assertLess(info["matrix_field_residual_difference"], 1e-7)

    def test_local_hidden_gradient_through_strain(self):
        model = EnergyNet(Case(geometry="cross"), width=3,
                          local_neural={"radius": .075, "width": 4})
        with torch.no_grad():
            model.local.coefficients.normal_(std=.02)
        quad = list(quadrature(model.cells, 8, 10))
        quad[0].requires_grad_(True)
        parameter = model.local.w1
        direction = torch.randn_like(parameter)
        energy = .5*energy_residual(model, quad).square().sum()
        gradient, = torch.autograd.grad(energy, parameter)
        original = parameter.detach().clone()
        values = []
        for step in (1e-5, -1e-5):
            with torch.no_grad():
                parameter.copy_(original+step*direction)
            values.append(.5*energy_residual(model, quad).square().sum().detach())
        with torch.no_grad():
            parameter.copy_(original)
        torch.testing.assert_close((values[0]-values[1])/2e-5,
                                   (gradient*direction).sum(), atol=1e-8, rtol=1e-6)

    def test_local_adaptation_keeps_inherited_field_and_reconstructs_energy(self):
        model = EnergyNet(Case(geometry="cross"), width=3,
                          local_neural={"radius": .075, "width": 4})
        with torch.no_grad():
            model.local.coefficients.normal_(std=.02)
        inherited = {name: p.detach().clone() for name, p in model.named_parameters()
                     if not name.startswith("local.")}
        hidden = model.local.w1.detach().clone()
        info = adapt_local_features(model, 8, 10, 8)
        self.assertLess(info["energy_after"], info["energy_before"])
        self.assertGreater(float((model.local.w1.detach()-hidden).abs().max()), 1e-8)
        for name, p in model.named_parameters():
            if name in inherited:
                torch.testing.assert_close(p, inherited[name], atol=0., rtol=0.)
        quad = list(quadrature(model.cells, 8, 10))
        quad[0].requires_grad_(True)
        actual = float(.5*energy_residual(model, quad).square().sum().detach())
        self.assertAlmostEqual(actual, info["energy_after"], places=12)


if __name__ == "__main__":
    unittest.main()
