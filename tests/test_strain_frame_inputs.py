"""Tests for the parity-matched strain-tensor inputs.

The four Tier-2 rotational magnitudes cannot represent the agyrotropic block:
under ``b -> -b`` the targets ``P12`` and ``P2par`` flip sign while every
magnitude is even, so the loss-minimising output for them is zero.  Feeding the
six Cartesian components of W *rotated into the model's own frame* fixes that,
because each in-frame strain component then carries the same parity as the
pressure component it drives.  These tests pin that property, the equivariance
it must not break, and the export path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from closure.field_invariants import (
    STRAIN_TENSOR_INDEX_PAIRS,
    STRAIN_TENSOR_NAMES,
    flow_strain_components,
    strain_tensor,
)
from closure.models import InvariantFieldAlignedPressureMLP

STRAIN_IDX = [10, 11, 12, 13, 14, 15]


def build(products: bool = False, **kw):
    dim = 20 if products else 8
    return InvariantFieldAlignedPressureMLP(
        feature_dims=[dim, 32, 6],
        activations=["SiLU", None],
        dropouts=[0.0, 0.0],
        use_electron_frame_invariants=False,
        strain_tensor_indices=STRAIN_IDX,
        strain_frame_scale=1.0,
        strain_frame_products=products,
        guide_direction=[0.0, 1.0, 0.0],
        **kw,
    ).eval()


def sample(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, generator=g, dtype=torch.float64)
    x[:, 0] = x[:, 0].abs() + 0.5          # density must be non-degenerate
    x[:, 1:4] += 0.3                        # keep |B| away from zero
    return x


def in_frame(net, pixels, packed_idx):
    """Rotate a packed Cartesian tensor from `pixels` into the model's frame."""
    rot, _ = net._basis_and_invariants(pixels)
    t = net._packed_to_tensor(pixels[:, packed_idx])
    return net._tensor_to_packed(torch.bmm(rot, torch.bmm(t, rot.transpose(1, 2))))


def test_strain_inputs_share_the_parity_of_the_pressure_targets():
    """W12/W1par/W2par must flip exactly as P12/P1par/P2par do under b -> -b."""
    net = build().double()
    x = sample()
    # An arbitrary symmetric "pressure" carried in the same packed layout.
    p_idx = [4, 5, 6, 7, 8, 9]

    flipped = x.clone()
    flipped[:, 1:4] *= -1.0

    w_a, w_b = in_frame(net, x, STRAIN_IDX), in_frame(net, flipped, STRAIN_IDX)
    p_a, p_b = in_frame(net, x, p_idx), in_frame(net, flipped, p_idx)

    # component order is [11, 22, parpar, 12, 1par, 2par]
    for k in range(6):
        sw = torch.sign((w_a[:, k] * w_b[:, k]).mean())
        sp = torch.sign((p_a[:, k] * p_b[:, k]).mean())
        assert sw == sp, f"parity mismatch in component {k}: strain {sw}, pressure {sp}"
    # and specifically: the agyrotropic pair really is odd
    assert torch.allclose(w_a[:, 3], -w_b[:, 3], atol=1e-10)   # W12   odd
    assert torch.allclose(w_a[:, 4], +w_b[:, 4], atol=1e-10)   # W1par even
    assert torch.allclose(w_a[:, 5], -w_b[:, 5], atol=1e-10)   # W2par odd


def test_equivariant_under_rotation_about_the_guide_axis():
    """The gauge is lab-fixed, so equivariance holds about the guide axis."""
    net = build().double()
    x = sample()
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    Q = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)

    xr = x.clone()
    xr[:, 1:4] = x[:, 1:4] @ Q.T
    xr[:, 4:7] = x[:, 4:7] @ Q.T
    w = net._packed_to_tensor(x[:, STRAIN_IDX])
    xr[:, STRAIN_IDX] = net._tensor_to_packed(Q @ w @ Q.T)

    with torch.no_grad():
        out = net._packed_to_tensor(net(x))
        out_r = net._packed_to_tensor(net(xr))
    assert torch.allclose(out_r, Q @ out @ Q.T, atol=1e-9)


def test_output_is_spd():
    net = build(products=True).double()
    with torch.no_grad():
        p = net._packed_to_tensor(net(sample()))
    assert torch.linalg.eigvalsh(p).min() > 0.0


def test_electric_field_is_still_ignored():
    """The variant must stay deployable: no E dependence at all."""
    net = build().double()
    x = sample()
    bumped = x.clone()
    bumped[:, 7:10] += 10.0
    with torch.no_grad():
        assert torch.equal(net(x), net(bumped))


@pytest.mark.parametrize("products", [False, True])
def test_torchscript_export(products):
    net = build(products=products)
    scripted = torch.jit.script(net)
    x = sample().float()
    with torch.no_grad():
        assert torch.allclose(scripted(x), net(x), atol=1e-6)


def test_reader_components_match_the_strain_tensor():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(3, 24, 20))
    dx, dy = 0.05, 0.05
    W, _ = strain_tensor(u, dx, dy)
    comps = flow_strain_components(u, dx, dy)
    assert tuple(comps) == STRAIN_TENSOR_NAMES
    for name, (i, j) in zip(STRAIN_TENSOR_NAMES, STRAIN_TENSOR_INDEX_PAIRS):
        assert np.allclose(comps[name], W[i, j])


def test_feature_dims_guard():
    with pytest.raises(ValueError):
        InvariantFieldAlignedPressureMLP(
            feature_dims=[6, 32, 6], activations=["SiLU", None], dropouts=[0.0, 0.0],
            use_electron_frame_invariants=False, strain_tensor_indices=STRAIN_IDX,
        )
    with pytest.raises(ValueError):
        InvariantFieldAlignedPressureMLP(
            feature_dims=[8, 32, 6], activations=["SiLU", None], dropouts=[0.0, 0.0],
            use_electron_frame_invariants=False, strain_tensor_indices=[10, 11, 12],
        )


# --------------------------------------------------------------------------
# Block-weighted loss
# --------------------------------------------------------------------------

SIGMAS = [6.772201e-03, 5.167816e-05, 7.994122e-05]   # training-split block rms


def build_block(lam):
    return InvariantFieldAlignedPressureMLP(
        feature_dims=[8, 32, 6], activations=["SiLU", None], dropouts=[0.0, 0.0],
        use_electron_frame_invariants=False, strain_tensor_indices=STRAIN_IDX,
        guide_direction=[0.0, 1.0, 0.0],
        block_loss_lambda=lam, block_loss_sigmas=SIGMAS,
    ).double().eval()


def test_blocks_partition_the_frobenius_norm():
    """The three block norms must sum to ||dP||_F^2 exactly, or lambda=0 shifts."""
    net = build_block(0.0)
    d = torch.randn(500, 6, dtype=torch.float64)
    g, m2, m1 = net._irreducible_blocks(d)
    total = (g ** 2).sum(1) + (m2 ** 2).sum(1) + (m1 ** 2).sum(1)
    frob = d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2 + 2 * (d[:, 3:] ** 2).sum(1)
    assert torch.allclose(total, frob, atol=1e-12)


def test_lambda_zero_reproduces_the_frobenius_loss():
    x, crit = sample().double(), torch.nn.MSELoss()
    pred = torch.randn(64, 6, dtype=torch.float64) * 3e-3
    targ = torch.randn(64, 6, dtype=torch.float64) * 3e-3
    a = build_block(0.0).compute_training_loss(x, pred, targ, crit)   # criterion path
    b = build_block(1e-14).compute_training_loss(x, pred, targ, crit)  # block path
    assert torch.allclose(a, b, rtol=1e-12)


@pytest.mark.parametrize("lam", [0.25, 0.5, 1.0])
def test_block_loss_invariant_under_rotation_about_b(lam):
    """Rotating about b mixes components inside a block but must not move the loss.

    This is the property that forbids per-component weights: only whole-block
    weights are well defined.
    """
    net = build_block(lam)
    pred = torch.randn(400, 6, dtype=torch.float64) * 3e-3
    targ = torch.randn(400, 6, dtype=torch.float64) * 3e-3

    def spin(packed, phi):
        """Rotate a packed field-frame tensor by phi about the parallel axis."""
        t = net._packed_to_tensor(packed)
        c, s = np.cos(phi), np.sin(phi)
        R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                         dtype=torch.float64)
        return net._tensor_to_packed(R @ t @ R.T)

    def block_loss(p, t):
        w = net.block_loss_weight
        return sum(w[k] * ((bp - bt) ** 2).sum(1).mean() for k, (bp, bt) in
                   enumerate(zip(net._irreducible_blocks(p), net._irreducible_blocks(t))))

    base = block_loss(pred, targ)
    for phi in (0.3, 1.1, 2.7):
        spun = block_loss(spin(pred, phi), spin(targ, phi))
        assert torch.allclose(base, spun, rtol=1e-10), f"not invariant at phi={phi}"


def test_per_component_weights_would_not_be_invariant():
    """Guard the design choice: weighting P1par and P2par differently breaks it."""
    net = build_block(0.5)
    pred = torch.randn(400, 6, dtype=torch.float64) * 3e-3
    bad = torch.tensor([1.0, 1.0, 1.0, 1.0, 40.0, 90.0], dtype=torch.float64)
    t = net._packed_to_tensor(pred)
    c, s = np.cos(0.9), np.sin(0.9)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    spun = net._tensor_to_packed(R @ t @ R.T)
    assert not torch.allclose((bad * pred ** 2).sum(), (bad * spun ** 2).sum(), rtol=1e-6)


def test_block_loss_config_guards():
    with pytest.raises(ValueError):
        build_block(-0.1)
    with pytest.raises(ValueError):
        InvariantFieldAlignedPressureMLP(
            feature_dims=[8, 32, 6], activations=["SiLU", None], dropouts=[0.0, 0.0],
            use_electron_frame_invariants=False, strain_tensor_indices=STRAIN_IDX,
            block_loss_lambda=0.5,
        )
