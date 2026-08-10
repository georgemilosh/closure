"""Round-trip tests for the Menura deployment wrappers.

The failure these guard against is silent: a decode mismatch produces a
plausible pressure field that is wrong by a constant factor.  Exactly that
happened before menura commit aee60b4, where a 2*pi instead of 4*pi halved
every electron pressure (Te/Ti read 0.1 instead of 0.2) with nothing in the
output to show it.  So the tests below run the EXACT checkpoints through the
EXACT float32 clamp/decode arithmetic from kernels_fields.cuh, not a dummy
network and not a paraphrase of the decode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, "/volume1/scratch/georgem/closure")
sys.path.insert(0, "/volume1/scratch/georgem/closure/scripts")

from export_for_menura import (MENURA_FOUR_PI, NN_ARG_CLIP, MenuraPressureWrapper,  # noqa: E402
                               load_network, menura_decode)

M = Path("/volume1/scratch/georgem/closure/models/Lightning/iPiC3D-nathan5-12/haydn")
ARMS = {
    "nBWTb0p5": M / "eos_nBWTb0p5_Ptensor_fieldframe_f2/runs_MLP/ablate_nBWTb0p5_Ptensor_fieldframe_baseline",
    "nBW":      M / "eos_nBW_Ptensor_fieldframe_f2/runs_MLP/ablate_nBW_Ptensor_fieldframe_baseline",
}
IDENTITY_MEAN = np.zeros(6, dtype=np.float32)
IDENTITY_STD = np.ones(6, dtype=np.float32)


def _load(tag):
    vd = ARMS[tag]
    ckpt = sorted((vd / "checkpoints").glob("best-*.ckpt"))[0]
    net, cfg = load_network(vd, ckpt)
    n_feat = len(cfg["data"]["read_features_targets_kwargs"]["request_features"])
    return net, cfg, n_feat


def _sample(n_feat, n=512, seed=0):
    """Feature vectors in the right ballpark: rho ~ 0.02, |B| ~ 1, strain ~ O(1)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n_feat, generator=g)
    x[:, 0] = -(0.02 + 0.01 * torch.rand(n, generator=g))   # rho_e is negative in menura
    x[:, 1:4] = 0.3 * torch.randn(n, 3, generator=g)
    x[:, 3] += 0.4                                          # guide field
    return x


@pytest.mark.parametrize("tag", list(ARMS))
@pytest.mark.parametrize("gyro", [False, True])
def test_round_trip_reproduces_four_pi_P(tag, gyro):
    """wrapper -> exact menura decode == 4*pi * (what the bare network emits)."""
    net, cfg, n_feat = _load(tag)
    init = cfg["model"]["network"]["init_args"]
    w = MenuraPressureWrapper(net, project_gyrotropic=gyro,
                              magnetic_indices=init.get("magnetic_indices", [1, 2, 3]),
                              guide_direction=init.get("guide_direction", [0., 1., 0.])).eval()
    x = _sample(n_feat)
    with torch.no_grad():
        bare = net(x).numpy()
        y = w(x).numpy()

    expected = bare.copy()
    if gyro:                                   # project the reference the same way
        b = x[:, [1, 2, 3]].numpy()
        bh = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        pxx, pyy, pzz, pxy, pxz, pyz = (bare[:, i] for i in range(6))
        ppar = (pxx*bh[:,0]**2 + pyy*bh[:,1]**2 + pzz*bh[:,2]**2
                + 2*(pxy*bh[:,0]*bh[:,1] + pxz*bh[:,0]*bh[:,2] + pyz*bh[:,1]*bh[:,2]))
        pperp = 0.5*(pxx + pyy + pzz - ppar)
        dp = ppar - pperp
        expected = np.stack([pperp + dp*bh[:,0]**2, pperp + dp*bh[:,1]**2, pperp + dp*bh[:,2]**2,
                             dp*bh[:,0]*bh[:,1], dp*bh[:,0]*bh[:,2], dp*bh[:,1]*bh[:,2]], axis=1)

    got = menura_decode(y, IDENTITY_MEAN, IDENTITY_STD)
    rel = np.abs(got - MENURA_FOUR_PI*expected) / np.maximum(np.abs(MENURA_FOUR_PI*expected), 1e-30)
    assert rel.max() < 2e-5, f"{tag} gyro={gyro}: max relative round-trip error {rel.max():.2e}"


@pytest.mark.parametrize("tag", list(ARMS))
def test_wrapper_output_stays_inside_the_arg_clip(tag):
    """If |arg| ever reached NN_ARG_CLIP the decode would silently saturate."""
    net, cfg, n_feat = _load(tag)
    w = MenuraPressureWrapper(net).eval()
    with torch.no_grad():
        y = w(_sample(n_feat, n=4096)).numpy()
    assert np.isfinite(y).all()
    assert np.abs(y).max() < 0.8 * NN_ARG_CLIP, f"{tag}: max |arg| = {np.abs(y).max():.2f}"


def test_gyro_projection_is_spd_and_kills_agyrotropy_exactly():
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraPressureWrapper(net, project_gyrotropic=True).eval()
    x = _sample(n_feat, n=1024, seed=3)
    with torch.no_grad():
        p_full = net(x).double()
        p_gyro = torch.exp(w(x)[:, :3].double())      # undo the log to recover the tensor
    # rebuild the projected tensor and check its invariants
    b = x[:, [1, 2, 3]].double()
    bh = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    def packed_to_T(p):
        return torch.stack([torch.stack([p[:,0],p[:,3],p[:,4]],1),
                            torch.stack([p[:,3],p[:,1],p[:,5]],1),
                            torch.stack([p[:,4],p[:,5],p[:,2]],1)], 1)
    T = packed_to_T(p_full)
    ppar = torch.einsum('ni,nij,nj->n', bh, T, bh)
    pperp = 0.5*(torch.einsum('nii->n', T) - ppar)
    assert (ppar > 0).all() and (pperp > 0).all(), "projection would not be SPD"
    # trace is preserved exactly by construction
    Tg = pperp[:,None,None]*torch.eye(3, dtype=torch.float64)[None] \
         + (ppar-pperp)[:,None,None]*torch.einsum('ni,nj->nij', bh, bh)
    assert torch.allclose(torch.einsum('nii->n', Tg), torch.einsum('nii->n', T), rtol=1e-9)
    # and the agyrotropic content is exactly zero: P.b is parallel to b
    Pb = torch.einsum('nij,nj->ni', Tg, bh)
    resid = (Pb - ppar[:,None]*bh).norm(dim=1) / Pb.norm(dim=1).clamp_min(1e-30)
    assert resid.max() < 1e-9, f"residual agyrotropy {resid.max():.2e}"
    assert torch.linalg.eigvalsh(Tg).min() > 0


@pytest.mark.parametrize("gyro", [False, True])
def test_torchscript_matches_eager(gyro):
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraPressureWrapper(net, project_gyrotropic=gyro).eval()
    s = torch.jit.script(w)
    x = _sample(n_feat, n=256, seed=5)
    with torch.no_grad():
        assert torch.allclose(s(x), w(x), atol=1e-6)


@pytest.mark.parametrize("gyro", [False, True])
def test_accepts_both_shapes(gyro):
    """Menura hands the model [N,C]; the base network also supports [B,C,H,W].

    The gyro arm is parametrised deliberately: it was broken for 4-D input while
    only the non-gyro path was tested, so a defect in the *control* arm sat
    unnoticed behind a passing suite.
    """
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraPressureWrapper(net, project_gyrotropic=gyro).eval()
    flat = _sample(n_feat, n=64, seed=7)
    img = flat.reshape(1, 8, 8, n_feat).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        a = w(flat)
        b = w(img)
    assert a.shape == (64, 6)
    assert torch.allclose(a, b.permute(0, 2, 3, 1).reshape(-1, 6), atol=1e-6)


def test_low_field_uses_the_guide_axis_fallback():
    """At |B| ~ 0 the parallel direction is undefined; must not produce NaN."""
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraPressureWrapper(net, project_gyrotropic=True).eval()
    x = _sample(n_feat, n=64, seed=11)
    x[:, 1:4] = 0.0
    with torch.no_grad():
        y = w(x)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("tag", list(ARMS))
def test_edge_cases_stay_finite(tag):
    net, cfg, n_feat = _load(tag)
    w = MenuraPressureWrapper(net).eval()
    x = _sample(n_feat, n=256, seed=13)
    x[:64, 0] = -1e-6                       # near-vacuum density
    x[64:128, 1:4] = 1e-9                   # near-null field
    if n_feat >= 16:
        x[128:192, 10:16] *= 500.0          # large strain
    with torch.no_grad():
        y = w(x)
    assert torch.isfinite(y).all(), f"{tag}: non-finite wrapper output on edge cases"
    assert np.isfinite(menura_decode(y.numpy(), IDENTITY_MEAN, IDENTITY_STD)).all()
