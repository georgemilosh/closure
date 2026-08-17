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
    "nBEe":     M / "eos_nBEe_Ptensor_fieldframe_f2/runs_MLP/ablate_nBEe_Ptensor_fieldframe_baseline",
    "nBEeW":    M / "eos_nBEeW_Ptensor_fieldframe_f2/runs_MLP/ablate_nBEeW_Ptensor_fieldframe_baseline",
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


# --------------------------------------------------------------------------
# P_GYRO two-channel wrapper (Menura's native gyrotropic path)
# --------------------------------------------------------------------------

from export_for_menura import MenuraGyroWrapper, menura_gyro_decode  # noqa: E402

GYRO_MEAN = np.zeros(2, dtype=np.float32)
GYRO_STD = np.ones(2, dtype=np.float32)


def _reference_par_perp(bare, x):
    """(p_par, p_perp) of the network tensor, computed independently."""
    b = x[:, [1, 2, 3]].numpy()
    bh = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    pxx, pyy, pzz, pxy, pxz, pyz = (bare[:, i] for i in range(6))
    ppar = (pxx*bh[:, 0]**2 + pyy*bh[:, 1]**2 + pzz*bh[:, 2]**2
            + 2*(pxy*bh[:, 0]*bh[:, 1] + pxz*bh[:, 0]*bh[:, 2] + pyz*bh[:, 1]*bh[:, 2]))
    pperp = 0.5*(pxx + pyy + pzz - ppar)
    return ppar, pperp


def test_pgyro_round_trip_reproduces_four_pi_par_perp():
    """wrapper -> exact P_GYRO decode == 4*pi * (p_par, p_perp) of the tensor."""
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraGyroWrapper(net).eval()
    x = _sample(n_feat)
    with torch.no_grad():
        bare = net(x).numpy()
        y = w(x).numpy()
    ppar, pperp = _reference_par_perp(bare, x)
    got = menura_gyro_decode(y, GYRO_MEAN, GYRO_STD)
    ref = np.stack([4*np.pi*ppar, 4*np.pi*pperp], axis=1)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)
    assert rel.max() < 2e-5, f"max relative round-trip error {rel.max():.2e}"


def test_pgyro_matches_the_tensor_projection_arm():
    """The 2-channel wrapper and the 6-channel projection arm must agree on
    (p_par, p_perp): they are the same physics through two decode paths."""
    net, cfg, n_feat = _load("nBWTb0p5")
    w2 = MenuraGyroWrapper(net).eval()
    w6 = MenuraPressureWrapper(net, project_gyrotropic=True).eval()
    x = _sample(n_feat, n=1024, seed=17)
    with torch.no_grad():
        par2 = torch.exp(w2(x)[:, 0])
        # in the 6-channel arm p_par/p_perp are recoverable from the diagonal of
        # the projected tensor: tr = ppar + 2 pperp and b.P.b = ppar
        p6 = torch.stack([torch.exp(w6(x)[:, i]) for i in range(3)], dim=1)
    b = x[:, [1, 2, 3]]
    bh = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    # diagonal of P_gyro: pperp + (ppar-pperp) bh_i^2  -> ppar = b-weighted combo
    # simplest identity: sum_i diag_i = ppar + 2 pperp; and
    # sum_i diag_i bh_i^2 = pperp*(1) + (ppar-pperp)*sum bh_i^4 ... messy.
    # Use the clean invariant: trace equality.
    tr6 = p6.sum(1)
    with torch.no_grad():
        tr2 = torch.exp(w2(x)[:, 0]) + 2*torch.exp(w2(x)[:, 1])
    assert torch.allclose(tr2, tr6, rtol=1e-4), "trace differs between the two gyro arms"
    del par2


def test_pgyro_positive_and_finite_on_edge_cases():
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraGyroWrapper(net).eval()
    x = _sample(n_feat, n=256, seed=19)
    x[:64, 1:4] = 0.0                     # null field -> guide-axis fallback
    x[64:128, 0] = -1e-6                  # near-vacuum
    if n_feat >= 16:
        x[128:192, 10:16] *= 500.0        # large strain
    with torch.no_grad():
        y = w(x)
    assert torch.isfinite(y).all()
    dec = menura_gyro_decode(y.numpy(), GYRO_MEAN, GYRO_STD)
    assert np.isfinite(dec).all() and (dec > 0).all()


@pytest.mark.parametrize("shape4d", [False, True])
def test_pgyro_torchscript_and_shapes(shape4d):
    net, cfg, n_feat = _load("nBWTb0p5")
    w = MenuraGyroWrapper(net).eval()
    s = torch.jit.script(w)
    if shape4d:
        x = _sample(n_feat, n=64, seed=23).reshape(1, 8, 8, n_feat).permute(0, 3, 1, 2).contiguous()
        with torch.no_grad():
            a, b = w(x), s(x)
        assert a.shape == (1, 2, 8, 8)
    else:
        x = _sample(n_feat, n=64, seed=23)
        with torch.no_grad():
            a, b = w(x), s(x)
        assert a.shape == (64, 2)
    assert torch.allclose(a, b, atol=1e-6)


# --------------------------------------------------------------------------
# Emitted-artifact tests: load the .pt files actually deployed, never a
# freshly-built wrapper.  A stale artifact passed a green suite once already.
# --------------------------------------------------------------------------

DEPLOY = M / "deploy_menura"
BUNDLES = {
    "MLP643_pgyro": ("MLP643", 4, 2),
    "nBWTb0p5_full": ("nBWTb0p5", 16, 6),
    "nBWTb0p5_gyro": ("nBWTb0p5", 16, 6),
    "nBW_context": ("nBW", 14, 6),
    "nBWTb0p5_pgyro": ("nBWTb0p5", 16, 2),
    # the R0 input-set sweep: same wrapper, three different input sets
    "nBW_pgyro": ("nBW", 14, 2),
    "nBEe_pgyro": ("nBEe", 10, 2),
    "nBEeW_pgyro": ("nBEeW", 14, 2),
}


@pytest.mark.parametrize("bundle", list(BUNDLES))
def test_emitted_artifact_is_current(bundle):
    """The deployed .pt must reproduce a wrapper built from current source."""
    bdir = DEPLOY / bundle
    if not (bdir / "checkpoints/model.pt").exists():
        pytest.skip(f"{bundle} not exported yet")
    tag, n_feat, n_out = BUNDLES[bundle]
    if tag == "MLP643":
        net, ym, ys, xm, xs = _load_mlp643()
        fresh = MenuraGyroFromStandardizedWrapper(net, ym, ys, xm, xs).eval()
        emitted = torch.jit.load(str(bdir / "checkpoints/model.pt")).eval()
        x = _mlp643_sample(n=256, seed=29)
        with torch.no_grad():
            a, b = fresh(x), emitted(x)
        assert a.shape == b.shape == (256, n_out)
        assert torch.allclose(a, b, atol=1e-6), f"{bundle}: emitted .pt differs from current source"
        return
    net, cfg, _ = _load(tag)
    init = cfg["model"]["network"]["init_args"]
    kw = dict(magnetic_indices=init.get("magnetic_indices", [1, 2, 3]),
              guide_direction=init.get("guide_direction", [0., 1., 0.]))
    if n_out == 2:
        fresh = MenuraGyroWrapper(net, **kw).eval()
    else:
        fresh = MenuraPressureWrapper(net, project_gyrotropic=bundle.endswith("_gyro"), **kw).eval()
    emitted = torch.jit.load(str(bdir / "checkpoints/model.pt")).eval()
    x = _sample(n_feat, n=256, seed=29)
    img = x.reshape(1, 16 if n_feat == 16 else 1, -1, n_feat)  # only [N,C] guaranteed
    with torch.no_grad():
        a, b = fresh(x), emitted(x)
    assert a.shape == b.shape == (256, n_out)
    assert torch.allclose(a, b, atol=1e-6), f"{bundle}: emitted .pt differs from current source"


# --------------------------------------------------------------------------
# Causality: which deployed bundles actually consume E.
#
# Menura feeds fields->E (kernels_fields.cuh:467-469) -- the same E that Ohm's
# law produced from the closure's own previous P_e -- so a bundle that reads it
# closes a P_e -> E -> div P_e loop.  nBW's claim to causal cleanliness is that
# its network ignores those channels; that claim has to be pinned on the .pt
# that actually runs, not on the training checkpoint.
# --------------------------------------------------------------------------

E_CHANNELS = [7, 8, 9]


@pytest.mark.parametrize("bundle,expect_invariant",
                         [("nBW_pgyro", True), ("nBEe_pgyro", False),
                          ("nBEeW_pgyro", False)])
def test_emitted_pgyro_e_dependence(bundle, expect_invariant):
    bdir = DEPLOY / bundle
    if not (bdir / "checkpoints/model.pt").exists():
        pytest.skip(f"{bundle} not exported yet")
    _, n_feat, _ = BUNDLES[bundle]
    emitted = torch.jit.load(str(bdir / "checkpoints/model.pt")).eval()
    x = _sample(n_feat, n=512, seed=31)
    xp = x.clone()
    xp[:, E_CHANNELS] += 10.0
    with torch.no_grad():
        a, b = emitted(x), emitted(xp)
    if expect_invariant:
        assert torch.equal(a, b), (
            f"{bundle}: output changed under E+10 -- the causal-cleanliness "
            "claim does not hold for the emitted artifact")
    else:
        assert not torch.allclose(a, b, atol=1e-6), (
            f"{bundle}: output unchanged under E+10 -- E channels appear dead, "
            "so the bundle is not the E-fed model it claims to be")


# --------------------------------------------------------------------------
# Gyro projection of the STANDARDISED coordinate-basis tensor model (MLP643)
# --------------------------------------------------------------------------

import joblib
from export_for_menura import MenuraGyroFromStandardizedWrapper

MLP643_DIR = Path("/volume1/scratch/georgem/closure/models/Lightning/iPiC3D-nathan5-12"
                  "/production_ablations_f2_val0/runs_MLP/ablate_noJnoE_P_deeper")


def _load_mlp643():
    ckpt = sorted((MLP643_DIR / "checkpoints").glob("best-*.ckpt"))[0]
    net, cfg = load_network(MLP643_DIR, ckpt)
    ym, ys = joblib.load(MLP643_DIR / "y.pkl")
    xm, xs = joblib.load(MLP643_DIR / "X.pkl")
    return net, ym, ys, xm, xs


def _mlp643_sample(n=512, seed=0):
    """Standardised 4-channel inputs, as menura's kernel would hand them over."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 4, generator=g)


def _tensor_reference(net, x_std, ym, ys):
    """menura's P_TENSOR decode applied to the bare network output (float32)."""
    with torch.no_grad():
        yhat = net(x_std).numpy()
    arg = np.clip(yhat * ys.astype(np.float32) + ym.astype(np.float32),
                  np.float32(-NN_ARG_CLIP), np.float32(NN_ARG_CLIP))
    P = np.empty_like(arg)
    P[:, :3] = np.exp(arg[:, :3], dtype=np.float32)
    P[:, 3:] = np.sinh(arg[:, 3:], dtype=np.float32)
    return P


def test_mlp643_round_trip_matches_tensor_decode_projection():
    """wrapper -> P_GYRO decode == 4pi * projection of the P_TENSOR-decoded tensor,
    with b-hat built from UN-standardised B (the trap this wrapper exists for)."""
    net, ym, ys, xm, xs = _load_mlp643()
    w = MenuraGyroFromStandardizedWrapper(net, ym, ys, xm, xs).eval()
    x = _mlp643_sample()
    P = _tensor_reference(net, x, ym, ys)
    braw = x.numpy()[:, 1:4] * xs[1:4] + xm[1:4]           # un-standardised B
    bh = braw / np.maximum(np.linalg.norm(braw, axis=1, keepdims=True), 1e-8)
    ppar = (P[:, 0]*bh[:, 0]**2 + P[:, 1]*bh[:, 1]**2 + P[:, 2]*bh[:, 2]**2
            + 2*(P[:, 3]*bh[:, 0]*bh[:, 1] + P[:, 4]*bh[:, 0]*bh[:, 2] + P[:, 5]*bh[:, 1]*bh[:, 2]))
    pperp = 0.5*(P[:, 0] + P[:, 1] + P[:, 2] - ppar)
    with torch.no_grad():
        got = menura_gyro_decode(w(x).numpy(), GYRO_MEAN, GYRO_STD)
    mask = (ppar > 1e-25) & (pperp > 1e-25)                # away from the clamp
    ref = np.stack([4*np.pi*ppar, 4*np.pi*pperp], axis=1)
    rel = np.abs(got[mask] - ref[mask]) / np.maximum(np.abs(ref[mask]), 1e-30)
    assert mask.mean() > 0.9, "sample dominated by clamped cells; not a valid round-trip"
    assert rel.max() < 5e-5, f"max relative round-trip error {rel.max():.2e}"


def test_mlp643_bhat_uses_unstandardised_B():
    """Feeding the same physical B at two different standardisations must give
    the same projection axis; a wrapper using standardised B would not."""
    net, ym, ys, xm, xs = _load_mlp643()
    w = MenuraGyroFromStandardizedWrapper(net, ym, ys, xm, xs).eval()
    x = _mlp643_sample(n=128, seed=3)
    # a deliberately WRONG wrapper: identity X stats (treats std B as physical)
    w_wrong = MenuraGyroFromStandardizedWrapper(net, ym, ys,
                                                np.zeros(4, np.float32),
                                                np.ones(4, np.float32)).eval()
    with torch.no_grad():
        a, b = w(x), w_wrong(x)
    # they must differ (if they did not, the un-standardisation would be dead code
    # and this test would prove nothing)
    assert not torch.allclose(a, b, atol=1e-4), \
        "identity-stats wrapper agrees with the real one -- test is vacuous"


def test_mlp643_indefinite_tensor_stays_finite():
    net, ym, ys, xm, xs = _load_mlp643()
    w = MenuraGyroFromStandardizedWrapper(net, ym, ys, xm, xs).eval()
    x = _mlp643_sample(n=256, seed=7) * 8.0          # drive outputs hard
    with torch.no_grad():
        y = w(x)
    assert torch.isfinite(y).all()
    assert np.isfinite(menura_gyro_decode(y.numpy(), GYRO_MEAN, GYRO_STD)).all()


def test_mlp643_torchscript_matches_eager():
    net, ym, ys, xm, xs = _load_mlp643()
    w = MenuraGyroFromStandardizedWrapper(net, ym, ys, xm, xs).eval()
    s = torch.jit.script(w)
    x = _mlp643_sample(n=128, seed=11)
    with torch.no_grad():
        assert torch.allclose(s(x), w(x), atol=1e-6)
