"""Finite-energy Williams modes for the 270-degree grout junction.

The singular leading traction vanishes on the wedge faces. The complete field
is coupled through the finite-stiffness spring energy.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
import torch
from torch import nn


def wedge_displacement(x, y, exponent, symmetric, nu):
    """Unit-E plane-strain displacement from an Airy r**(lambda+1) mode.

    The material sector uses an unwrapped angle in [pi/2, 2*pi], centered on 5*pi/4.
    Points at the vertex are excluded because its derivatives are singular.
    """
    radius = (x.square()+y.square()).sqrt()
    angle = torch.atan2(y, x)
    angle = torch.where(angle <= 0., angle+2*np.pi, angle)
    theta = angle-5*np.pi/4
    alpha, lam = 3*np.pi/4, exponent
    p, q = lam+1, lam-1
    if symmetric:
        a, b = np.cos(q*alpha), -np.cos(p*alpha)
        f = a*torch.cos(p*theta)+b*torch.cos(q*theta)
        df = -p*a*torch.sin(p*theta)-q*b*torch.sin(q*theta)
        ddf = -p*p*a*torch.cos(p*theta)-q*q*b*torch.cos(q*theta)
        dddf = p**3*a*torch.sin(p*theta)+q**3*b*torch.sin(q*theta)
    else:
        a, b = np.sin(q*alpha), -np.sin(p*alpha)
        f = a*torch.sin(p*theta)+b*torch.sin(q*theta)
        df = p*a*torch.cos(p*theta)+q*b*torch.cos(q*theta)
        ddf = -p*p*a*torch.sin(p*theta)-q*q*b*torch.sin(q*theta)
        dddf = -p**3*a*torch.cos(p*theta)-q**3*b*torch.cos(q*theta)
    factor = (1+nu)/lam
    ur = factor*((1-nu)*ddf+p*(1-nu-nu*lam)*f)
    dur = factor*((1-nu)*dddf+p*(1-nu-nu*lam)*df)
    ut = (-2*(1+nu)*lam*df-dur)/(lam-1)
    c, s = x/radius, y/radius
    return radius[..., None]**lam*torch.stack([ur*c-ut*s, ur*s+ut*c], -1)


class JunctionModes(nn.Module):
    """Material-shared local modes with reflected symmetry and compact support."""
    def __init__(self, case, cells, half_width, radius):
        super().__init__()
        if case.geometry != "cross":
            raise ValueError("Junction modes require the cross geometry.")
        if not 0. < radius < min(.5-half_width, case.vertical_span/2-half_width,
                                case.horizontal_span/2-half_width):
            raise ValueError("Junction support must stay clear of the outer and tip boundaries.")
        self.radius, self.half_width, self.nu = radius, half_width, case.poisson
        self.exponents = (brentq(lambda z: np.sin(1.5*np.pi*z)-z, .5, .6),
                          brentq(lambda z: np.sin(1.5*np.pi*z)+z, .8, .95))
        self.coefficients = nn.Parameter(torch.zeros(2))
        self.register_buffer("grout_mask", torch.tensor(
            [c["material"] == "grout" for c in cells])[:, None, None, None])
        self.scale = 1/np.sqrt(case.grout_E_MPa/case.soil_E_MPa)

    def basis(self, xy):
        result = torch.zeros((*xy.shape[:2], 2, 2))
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            x = sx*xy[..., 0]-self.half_width
            y = sy*xy[..., 1]-self.half_width
            window = torch.relu(1-(x.square()+y.square())/self.radius**2)**3
            modes = torch.stack([wedge_displacement(x, y, lam, j == 0, self.nu)
                                 for j, lam in enumerate(self.exponents)], -2)
            result = result+window[..., None, None]*modes*torch.tensor([sx, sy])
        return result*self.grout_mask*self.scale

    def forward(self, xy):
        return (self.basis(xy)*self.coefficients[:, None]).sum(-2)
