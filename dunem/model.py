"""DUNEM-PET network (inference).

Per plane, the network unrolls K iterations on the objective

    Phi_eps(x, y) = Gamma_eps(G x - y) + beta * Omega(y; s) + Lambda_eps(x) + Psi_eps(y)

where Lambda (image domain), Psi (projection domain) and Gamma (residual coupling) are
learned Huber-smoothed group-sparsity potentials and Omega is the Poisson term. Each
iteration updates the sinogram block (y) and then the image block (x) with learned,
per-iteration step multipliers, accepts the update under a sufficient-descent test and
otherwise takes a safeguarded block-coordinate step, then applies the smoothing
continuation. Iterations are grouped into stages; every stage has its own potentials.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- #
# smoothed ReLU and its derivative
# --------------------------------------------------------------------------- #
class SmoothReLU(nn.Module):
    """psi(t)  = 0 | t^2/(4a) + t/2 + a/4 | t          (|t| vs a)     shift=False
       psi0(t) = psi(t - a) = 0 | t^2/(4a) | t - a     (t vs 0, 2a)   shift=True"""

    def __init__(self, a: float, shift: bool = False):
        super().__init__()
        self.a = float(a)
        self.shift = bool(shift)

    def forward(self, t):
        if self.shift:
            return torch.where(
                torch.le(t, 0.0), torch.zeros_like(t),
                torch.where(torch.ge(t, 2 * self.a), t - self.a, t * t / (4 * self.a)))
        return torch.where(
            torch.gt(torch.abs(t), self.a), torch.relu(t),
            t * t / (4 * self.a) + 0.5 * t + 0.25 * self.a)


class SmoothReLUDeriv(nn.Module):
    """d psi / dt, with the same shift as SmoothReLU."""

    def __init__(self, a: float, shift: bool = False):
        super().__init__()
        self.a = float(a)
        self.shift = bool(shift)

    def forward(self, t):
        if self.shift:
            t = t - self.a
        step = torch.where(t > 0, torch.ones_like(t), torch.zeros_like(t))
        return torch.where(torch.gt(torch.abs(t), self.a), step, t / (2 * self.a) + 0.5)


# --------------------------------------------------------------------------- #
# learned potential  T_eps(v) = sum_p h_eps(||g_p(v)||)
# --------------------------------------------------------------------------- #
class SparseTerm(nn.Module):
    """g(v) = W4 * psi(W3 * psi(W2 * psi(W1 * v)))                 anchor = 0   (Lambda, Psi)
       g(v) = [ anchor*v ; W4 * psi0(...) ]                        anchor > 0   (Gamma)

    Four bias-free convolutions; t_p = ||g_p(v)|| over channels; h_eps is the Huber
    function. grad() is the analytic gradient dT/dv."""

    def __init__(self, h: int, w: int, in_ch: int, channels: int, kernel, padding,
                 a_smooth: float, n_convs: int = 4, anchor: float = 0.0,
                 act_shift: bool = False):
        super().__init__()
        self.h, self.w = int(h), int(w)
        self.in_ch = int(in_ch)
        self.padding = tuple(padding)
        self.act = SmoothReLU(a_smooth, shift=act_shift)
        self.deri = SmoothReLUDeriv(a_smooth, shift=act_shift)
        self.anchor = float(anchor)
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.in_ch, channels, kernel_size=kernel, padding=padding, bias=False)]
            + [nn.Conv2d(channels, channels, kernel_size=kernel, padding=padding, bias=False)
               for _ in range(n_convs - 1)])

    def _feat(self, h0) -> List[torch.Tensor]:
        cache, z = [], h0
        for i, conv in enumerate(self.convs):
            z = conv(z) if i == 0 else conv(self.act(z))
            cache.append(z)
        return cache

    def _tnorm(self, h0, c):
        sq = torch.square(c).sum(dim=1, keepdim=True)
        if self.anchor > 0:
            sq = sq + torch.square(self.anchor * h0)
        pos = sq > 0
        return torch.where(pos, torch.sqrt(torch.where(pos, sq, torch.ones_like(sq))),
                           torch.zeros_like(sq))

    def value(self, v, eps):
        h0 = v.view(-1, self.in_ch, self.h, self.w)
        t = self._tnorm(h0, self._feat(h0)[-1])
        hub = torch.where(torch.le(t, eps), torch.square(t) / (2 * eps), t - eps / 2)
        return hub.sum()

    def grad(self, v, eps):
        h0 = v.view(-1, self.in_ch, self.h, self.w)
        cache = self._feat(h0)
        t = self._tnorm(h0, cache[-1])
        denom = torch.where(torch.ge(t, eps), t, eps)
        out = cache[-1] / denom
        for i in range(len(cache) - 1, 0, -1):
            out = F.conv_transpose2d(out, self.convs[i].weight,
                                     padding=self.padding) * self.deri(cache[i - 1])
        out = F.conv_transpose2d(out, self.convs[0].weight, padding=self.padding)
        if self.anchor > 0:
            out = out + (self.anchor ** 2) * (h0 / denom)
        return out.view(1, self.in_ch * self.h * self.w, 1)


# --------------------------------------------------------------------------- #
# Poisson term  Omega(y; s) = 1^T y - s^T log y  (log smoothed below tau)
# --------------------------------------------------------------------------- #
class PoissonTerm(nn.Module):
    """ell_tau(y) = log y for y >= tau, its 2nd-order Taylor extension below tau, plus a
    one-sided quadratic (1/2tau) max(0, tau - y)^2. tau = tau_frac * one photon."""

    def __init__(self, tau_frac: float = 0.1, floor: float = 1e-12):
        super().__init__()
        self.tau_frac = float(tau_frac)
        self.floor = float(floor)
        self.tau = None

    def photon(self, s) -> float:
        pos = s[s > 0]
        return float(pos.min()) if pos.numel() else 1.0

    def set_plane(self, s):
        self.tau = self.tau_frac * self.photon(s)

    def init_y(self, s):
        """y0 = max(s, tau)."""
        return torch.clamp_min(s, self.tau_frac * self.photon(s))

    def _ell(self, y):
        t = self.tau
        d = y - t
        return torch.where(y >= t,
                           torch.log(y.clamp_min(self.floor)),
                           math.log(t) + d / t - d * d / (2 * t * t))

    def value(self, y, s):
        t = self.tau
        return (y.sum() - (s * self._ell(y)).sum()
                + torch.clamp(t - y, min=0.0).pow(2).sum() / (2 * t))

    def grad(self, y, s):
        t = self.tau
        d = y - t
        dell = torch.where(y >= t, 1.0 / y.clamp_min(self.floor), 1.0 / t - d / (t * t))
        return 1.0 - s * dell - torch.clamp(t - y, min=0.0) / t


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
class TermGroup(nn.Module):
    """Lambda (R), Psi (Q) and Gamma of one stage."""

    def __init__(self, R: SparseTerm, Q: SparseTerm, Gamma: SparseTerm):
        super().__init__()
        self.R, self.Q, self.Gamma = R, Q, Gamma


class Objective(nn.Module):
    """Phi_eps(x, y) = Gamma_eps(Gx - y) + lam * Omega(y; s) + R_eps(x) + Q_eps(y)."""

    def __init__(self, groups, lam: float, eps0_R: float, eps0_Q: float, eps0_G: float,
                 tau_frac: float = 0.1):
        super().__init__()
        self.groups = nn.ModuleList(groups)
        self.active = 0
        self.Omega = PoissonTerm(tau_frac=tau_frac)
        self.lam = float(lam)
        self.eps0_R, self.eps0_Q, self.eps0_G = float(eps0_R), float(eps0_Q), float(eps0_G)
        self.op = None
        self.s = None

    @property
    def R(self):
        return self.groups[self.active].R

    @property
    def Q(self):
        return self.groups[self.active].Q

    @property
    def Gamma(self):
        return self.groups[self.active].Gamma

    def set_group(self, g: int):
        self.active = int(g)

    def set_plane(self, op, s):
        self.op, self.s = op, s
        self.Omega.set_plane(s)

    def eps(self, eps_mul):
        return eps_mul * self.eps0_R, eps_mul * self.eps0_Q, eps_mul * self.eps0_G

    def resid(self, x, y):
        return self.op.fwd(x) - y

    def f_value(self, x, y, eps_mul):
        _, _, eG = self.eps(eps_mul)
        return self.Gamma.value(self.resid(x, y), eG) + self.lam * self.Omega.value(y, self.s)

    def grad_f_x(self, x, y, eps_mul):
        _, _, eG = self.eps(eps_mul)
        return self.op.adj(self.Gamma.grad(self.resid(x, y), eG))

    def grad_f_y(self, x, y, eps_mul):
        _, _, eG = self.eps(eps_mul)
        return -self.Gamma.grad(self.resid(x, y), eG) + self.lam * self.Omega.grad(y, self.s)

    def value(self, x, y, eps_mul):
        eR, eQ, _ = self.eps(eps_mul)
        return self.f_value(x, y, eps_mul) + self.R.value(x, eR) + self.Q.value(y, eQ)

    def grad_x(self, x, y, eps_mul):
        eR, _, _ = self.eps(eps_mul)
        return self.grad_f_x(x, y, eps_mul) + self.R.grad(x, eR)

    def grad_y(self, x, y, eps_mul):
        _, eQ, _ = self.eps(eps_mul)
        return self.grad_f_y(x, y, eps_mul) + self.Q.grad(y, eQ)

    def grad_norm(self, x, y, eps_mul):
        gx, gy = self.grad_x(x, y, eps_mul), self.grad_y(x, y, eps_mul)
        return torch.sqrt(gx.pow(2).sum() + gy.pow(2).sum())


# --------------------------------------------------------------------------- #
# per-iteration step sizes  s = m_{k,i} * min(p ||v|| / max(||g||, rho), xi * S_i)
# --------------------------------------------------------------------------- #
class StepSizes(nn.Module):
    NAMES = ("alpha", "alpha_hat", "beta", "beta_hat")

    def __init__(self, n_phases: int, n_groups: int, p: float):
        super().__init__()
        self.p = float(p)
        self.register_buffer("caps", torch.zeros(int(n_groups), 4))
        self.log_m = nn.Parameter(torch.zeros(int(n_phases), 4))

    def base(self, i: int, v, g, eps_mul: float = 1.0, grp: int = 0):
        gn = g.norm()
        vn = v.norm()
        rel = self.p * vn / gn.clamp_min(1e-30)
        cap = eps_mul * float(self.caps[grp, i])
        return torch.clamp(rel, max=cap)

    def step(self, i: int, k: int, v, g, eps_mul: float = 1.0, grp: int = 0):
        return torch.exp(self.log_m[k, i]) * self.base(i, v, g, eps_mul, grp)


# --------------------------------------------------------------------------- #
# DUNEM-PET
# --------------------------------------------------------------------------- #
class DUNEMPET(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        c = config
        H, W = c["img_hw"]
        Rr, V = c["sino_hw"]
        self.group_bounds = [int(b) for b in c["group_bounds"]]
        self.n_phases = int(c["n_phases"])
        n_groups = len(self.group_bounds) + 1

        def make_group():
            R = SparseTerm(H, W, 1, c["channels"], c["R_kernel"], c["R_padding"],
                           c["a_smooth"], n_convs=c["n_convs"], anchor=0.0, act_shift=False)
            Q = SparseTerm(Rr, V, 1, c["channels"], c["S_kernel"], c["S_padding"],
                           c["a_smooth"], n_convs=c["n_convs"], anchor=0.0, act_shift=False)
            G = SparseTerm(Rr, V, 1, c["channels"], c["S_kernel"], c["S_padding"],
                           c["a_smooth"], n_convs=c["n_convs"],
                           anchor=float(c["eps0_G"]) ** 0.5, act_shift=True)
            return TermGroup(R, Q, G)

        self.phi = Objective([make_group() for _ in range(n_groups)], lam=c["lam"],
                             eps0_R=c["eps0_R"], eps0_Q=c["eps0_Q"], eps0_G=c["eps0_G"],
                             tau_frac=c["tau_frac"])
        self.steps = StepSizes(self.n_phases, n_groups, p=c["step_p"])
        self.eps_decay = float(c["eps_decay"])
        self.anneal_scale = float(c["anneal_scale"])
        self.sdc_eta = float(c["sdc_eta"])
        self.bcd_delta = float(c["bcd_delta"])
        self.bcd_rho = float(c["bcd_rho"])
        self.bcd_max_retry = int(c["bcd_max_retry"])

    def init_y(self, s):
        return self.phi.Omega.init_y(s)

    def group_of(self, k: int) -> int:
        g = 0
        for b in self.group_bounds:
            if k >= b:
                g += 1
        return g

    # ---- one iteration: sinogram block, then image block -------------------- #
    def _phase(self, x, y, eps_mul, k, grp: int = 0):
        self.phi.set_group(grp)
        eR, eQ, _ = self.phi.eps(eps_mul)
        phi0 = self.phi.value(x, y, eps_mul)
        gfy = self.phi.grad_f_y(x, y, eps_mul)
        b = y - self.steps.step(0, k, y, gfy, eps_mul, grp) * gfy
        gq = self.phi.Q.grad(b, eQ)
        u_y = b - self.steps.step(1, k, b, gq, eps_mul, grp) * gq
        gfx = self.phi.grad_f_x(x, u_y, eps_mul)
        c = x - self.steps.step(2, k, x, gfx, eps_mul, grp) * gfx
        gr = self.phi.R.grad(c, eR)
        u_x = c - self.steps.step(3, k, c, gr, eps_mul, grp) * gr
        return phi0, gfy, u_x, u_y

    # ---- sufficient-descent test ------------------------------------------- #
    def _sdc(self, x, y, u_x, u_y, phi0, grad_norm, eps_mul):
        dx, dy = u_x - x, u_y - y
        sq = dx.pow(2).sum() + dy.pow(2).sum()
        ok_a = (self.phi.value(u_x, u_y, eps_mul) - phi0).item() <= (-self.sdc_eta * sq).item()
        ok_b = grad_norm.item() <= (dx.norm() + dy.norm()).item() / self.sdc_eta
        return ok_a and ok_b

    # ---- safeguarded block-coordinate step with backtracking --------------- #
    def _bcd(self, x, y, eps_mul, gfy, phi0, bcd_scale, grp: int = 0):
        self.phi.set_group(grp)
        eR, eQ, _ = self.phi.eps(eps_mul)
        gqy = self.phi.Q.grad(y, eQ)
        grx = self.phi.R.grad(x, eR)
        gy = gfy + gqy
        v_x = v_y = None
        for _ in range(self.bcd_max_retry):
            sy = bcd_scale * self.steps.base(0, y, gy, eps_mul, grp)
            v_y = y - sy * gy
            gx = self.phi.grad_f_x(x, v_y, eps_mul) + grx
            sx = bcd_scale * self.steps.base(2, x, gx, eps_mul, grp)
            v_x = x - sx * gx
            dvx, dvy = v_x - x, v_y - y
            sq = dvx.pow(2).sum() + dvy.pow(2).sum()
            if (self.phi.value(v_x, v_y, eps_mul) - phi0).item() <= (-self.bcd_delta * sq).item():
                return v_x, v_y, bcd_scale
            bcd_scale = self.bcd_rho * bcd_scale
        return v_x, v_y, bcd_scale

    # ---- reconstruction of one plane ---------------------------------------- #
    @torch.no_grad()
    def forward(self, x0, y0):
        """`phi.set_plane(op, s)` must have been called for this plane."""
        eps_mul = 1.0
        bcd_scale = 1.0
        gcache = None                       # (||grad Phi||, eps_mul it was computed at)
        x, y = x0, y0
        for k in range(self.n_phases):
            grp = self.group_of(k)
            self.phi.set_group(grp)
            if gcache is not None and gcache[1] == eps_mul:
                gnorm = gcache[0]
            else:
                gnorm = self.phi.grad_norm(x, y, eps_mul)

            phi0, gfy, u_x, u_y = self._phase(x, y, eps_mul, k, grp)

            if self._sdc(x, y, u_x, u_y, phi0, gnorm, eps_mul):
                nx, ny = u_x, u_y
            else:
                nx, ny, bcd_scale = self._bcd(x, y, eps_mul, gfy, phi0, bcd_scale, grp)

            # smoothing continuation: all three eps shrink together
            gnew = self.phi.grad_norm(nx, ny, eps_mul)
            if gnew.item() < self.anneal_scale * self.eps_decay * eps_mul * self.phi.eps0_G:
                eps_mul = eps_mul * self.eps_decay
                gcache = None
            else:
                gcache = (gnew, eps_mul)
            x, y = nx, ny
        return x, y


def load_model(path: str, device: str) -> DUNEMPET:
    ck = torch.load(path, map_location=device, weights_only=True)
    net = DUNEMPET(ck["config"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net
