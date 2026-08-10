"""Regression tests for the equilibrium-anchored residual closure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from closure.models import EquilibriumAnchoredResidualPressureMLP
from closure.module import ClosureLitModule
from closure.datamodule import _ResidualAnchorFileDataset

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from export_for_menura import (  # noqa: E402
    MENURA_FOUR_PI,
    MenuraPressureWrapper,
    menura_decode,
)


def residual_model(alpha: float = 0.25) -> EquilibriumAnchoredResidualPressureMLP:
    return EquilibriumAnchoredResidualPressureMLP(
        feature_dims=[8, 24, 16, 6],
        activations=["SiLU", "SiLU", None],
        dropouts=[0.0, 0.0, 0.0],
        strain_tensor_indices=[10, 11, 12, 13, 14, 15],
        strain_frame_scale=2.7,
        residual_alpha=alpha,
        block_loss_lambda=0.5,
        block_loss_sigmas=[0.006772201, 5.167816e-05, 7.994122e-05],
    )


def features(samples: int = 32) -> torch.Tensor:
    generator = torch.Generator().manual_seed(812)
    x = torch.randn(samples, 16, generator=generator)
    x[:, 0] = -(0.015 + 0.07 * torch.rand(samples, generator=generator))
    x[:, 1:4] = 0.25 * torch.randn(samples, 3, generator=generator)
    x[:, 1] += 0.8
    x[:, 3] += 0.4
    x[:, 4:10] *= 0.2
    x[:, 10:16] *= 2.0
    x[:, 12] = 0.0
    return x


def test_zero_residual_is_exact_analytic_equilibrium():
    model = residual_model()
    x = features()
    pixels, _, _, _ = model._as_pixels(x)
    rotation, _ = model._basis_and_invariants(pixels)
    base_field = model._analytic_base_field(pixels)
    expected = model._tensor_to_packed(
        rotation.transpose(1, 2) @ base_field @ rotation
    )
    actual = model(x)
    assert torch.allclose(actual, expected, rtol=1e-7, atol=4e-10)


def test_correction_parameters_obey_bounds():
    alpha = 0.25
    model = residual_model(alpha)
    raw = torch.tensor(
        [[1000.0, -1000.0, 50.0, 1000.0, -1000.0, 1000.0]], dtype=torch.float32
    )
    correction = model._correction_matrix(raw)
    diagonal = torch.diagonal(correction, dim1=1, dim2=2)
    lower = correction[:, [1, 2, 2], [0, 0, 1]]
    assert float(diagonal.min()) >= float(np.exp(-alpha)) * (1.0 - 1.0e-6)
    assert float(diagonal.max()) <= float(np.exp(alpha)) * (1.0 + 1.0e-6)
    assert lower.abs().max() <= alpha


def test_output_is_spd_for_large_finite_inputs():
    model = residual_model()
    # Exercise non-zero residuals, including saturated tanh corrections.
    final = [layer for layer in model.trunk.linear_relu_stack if isinstance(layer, torch.nn.Linear)][-1]
    torch.nn.init.normal_(final.weight, std=20.0)
    torch.nn.init.normal_(final.bias, std=20.0)
    tensor = model._packed_to_tensor(model(features(256)))
    eigenvalues = torch.linalg.eigvalsh(tensor)
    assert torch.isfinite(tensor).all()
    assert eigenvalues.min() > 0.0


def test_pixel_image_and_torchscript_parity():
    model = residual_model().eval()
    flat = features(16)
    image = flat.reshape(1, 4, 4, 16).permute(0, 3, 1, 2).contiguous()
    scripted = torch.jit.script(model)
    with torch.no_grad():
        flat_out = model(flat)
        image_out = model(image).permute(0, 2, 3, 1).reshape(-1, 6)
        script_out = scripted(flat)
    assert torch.allclose(flat_out, image_out, atol=1e-7)
    assert torch.allclose(flat_out, script_out, atol=1e-7)


def test_menura_wrapper_decode_reconstructs_four_pi_pressure():
    model = residual_model().eval()
    wrapper = MenuraPressureWrapper(model).eval()
    x = features(64)
    with torch.no_grad():
        pressure = model(x).numpy()
        arguments = wrapper(x).numpy()
    decoded = menura_decode(
        arguments, np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)
    )
    np.testing.assert_allclose(decoded, MENURA_FOUR_PI * pressure, rtol=2e-5, atol=1e-8)


def test_normalized_jvp_penalty_matches_linear_jacobian():
    network = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0]]))
    module = ClosureLitModule(
        network=network,
        scheduler=None,
        lambda_jacobian=1.0,
        jacobian_feature_indices=[1],
        jacobian_feature_scales=[2.0],
        jacobian_pressure_scale=4.0,
        jacobian_samples=8,
    ).eval()
    x = torch.randn(8, 3)
    actual = module._normalized_jvp_penalty(x, create_graph=False)
    # The selected direction is +/-2 in feature 1.  The two output tangents
    # are therefore +/-[4,8], divided by pressure scale 4 -> [1,2].
    expected = torch.tensor((1.0**2 + 2.0**2) / 2.0)
    assert torch.allclose(actual, expected)


def test_jvp_penalty_backpropagates_to_residual_weights():
    model = residual_model()
    # Break exact zero initialization so dP/dW depends on the trunk weights.
    final = [layer for layer in model.trunk.linear_relu_stack if isinstance(layer, torch.nn.Linear)][-1]
    torch.nn.init.normal_(final.weight, std=1.0e-2)
    module = ClosureLitModule(
        network=model,
        scheduler=None,
        lambda_jacobian=0.1,
        jacobian_feature_indices=[10, 11, 12, 13, 14, 15],
        jacobian_feature_scales=[2.7] * 6,
        jacobian_pressure_scale=0.003,
        jacobian_samples=16,
    ).train()
    penalty = module._normalized_jvp_penalty(features(32), create_graph=True)
    penalty.backward()
    assert penalty > 0.0
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()


def test_anchor_weight_does_not_capture_arbitrary_low_strain_kinetic_cell():
    model = residual_model()
    x = features(8)
    x[:, 10:16] = 0.0
    with torch.no_grad():
        equilibrium = model(x)
    # A kinetic target can differ from P0 even when W is small.  Its loss must
    # not receive the synthetic-anchor multiplier.
    kinetic = equilibrium.clone()
    kinetic[:, 0] += 0.1 * model.pressure_scale
    prediction = kinetic + 0.01 * model.pressure_scale
    weighted = model.compute_training_loss(x, prediction, kinetic, torch.nn.MSELoss())
    original_weight = model.anchor_loss_weight
    model.anchor_loss_weight = 1.0
    unweighted = model.compute_training_loss(x, prediction, kinetic, torch.nn.MSELoss())
    model.anchor_loss_weight = original_weight
    assert torch.allclose(weighted, unweighted, rtol=1e-6, atol=1e-10)


def test_exact_equilibrium_target_receives_anchor_weight():
    model = residual_model()
    x = features(1).repeat(8, 1)
    # Actual particle startup noise gives Harris anchors substantial W.  The
    # exact P0 target, not a quiet-input heuristic, must identify them.
    x[:, 10:16] = 2.0
    with torch.no_grad():
        target = model(x)
    # The last half are ordinary low-W kinetic cells, distinguished by a
    # non-equilibrium target rather than by changing the model input.
    target[4:, 0] += 0.1 * model.pressure_scale
    prediction = target.clone()
    prediction[0, 0] += 0.1 * model.pressure_scale
    weighted = model.compute_training_loss(x, prediction, target, torch.nn.MSELoss())
    original_weight = model.anchor_loss_weight
    model.anchor_loss_weight = 1.0
    unweighted = model.compute_training_loss(x, prediction, target, torch.nn.MSELoss())
    model.anchor_loss_weight = original_weight
    assert weighted > 1.5 * unweighted


def test_anchor_mask_is_stable_under_bfloat16_autocast():
    model = residual_model()
    x = features(8)
    x[:, 10:16] = 2.0
    with torch.no_grad():
        target = model(x)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pixels, _, _, _ = model._as_pixels(x)
        rotation, _ = model._basis_and_invariants(pixels)
        mask = model._equilibrium_anchor_mask(pixels, target, rotation)
    assert mask.all()
    kinetic = target.clone()
    kinetic[:, 0] += 0.01 * model.pressure_scale
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mask = model._equilibrium_anchor_mask(pixels, kinetic, rotation)
    assert not mask.any()


def _write_anchor_file(path: Path, *, feature_names=None, target_names=None):
    feature_names = feature_names or [
        "rho_e", "Bx", "By", "Bz", "Vx_e", "Vy_e", "Vz_e", "Ex", "Ey", "Ez",
        "Wxx_e", "Wyy_e", "Wzz_e", "Wxy_e", "Wxz_e", "Wyz_e",
    ]
    target_names = target_names or [
        "Pxx_e", "Pyy_e", "Pzz_e", "Pxy_e", "Pxz_e", "Pyz_e",
    ]
    np.savez(
        path,
        features=np.arange(48, dtype=np.float32).reshape(3, 16),
        targets=np.arange(18, dtype=np.float32).reshape(3, 6),
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
    )
    return feature_names, target_names


def test_residual_anchor_file_loads_exact_order_and_values(tmp_path):
    path = tmp_path / "anchors.npz"
    feature_names, target_names = _write_anchor_file(path)
    dataset = _ResidualAnchorFileDataset(
        str(path), feature_names=feature_names, target_names=target_names
    )
    assert len(dataset) == 3
    actual_features, actual_targets = dataset[2]
    torch.testing.assert_close(actual_features, torch.arange(32, 48, dtype=torch.float32))
    torch.testing.assert_close(actual_targets, torch.arange(12, 18, dtype=torch.float32))


def test_residual_anchor_file_rejects_channel_reordering(tmp_path):
    path = tmp_path / "anchors.npz"
    feature_names, target_names = _write_anchor_file(path)
    feature_names[0], feature_names[1] = feature_names[1], feature_names[0]
    with pytest.raises(ValueError, match="channel order mismatch"):
        _ResidualAnchorFileDataset(
            str(path), feature_names=feature_names, target_names=target_names
        )


@pytest.mark.parametrize(
    ("features", "targets", "message"),
    [
        (np.zeros((3, 15), dtype=np.float32), np.zeros((3, 6), dtype=np.float32), "feature shape"),
        (np.zeros((3, 16), dtype=np.float32), np.zeros((2, 6), dtype=np.float32), "target shape"),
        (np.full((3, 16), np.nan, dtype=np.float32), np.zeros((3, 6), dtype=np.float32), "nonfinite"),
    ],
)
def test_residual_anchor_file_rejects_bad_arrays(tmp_path, features, targets, message):
    path = tmp_path / "bad.npz"
    feature_names, target_names = _write_anchor_file(path)
    np.savez(
        path,
        features=features,
        targets=targets,
        feature_names=np.asarray(feature_names),
        target_names=np.asarray(target_names),
    )
    with pytest.raises(ValueError, match=message):
        _ResidualAnchorFileDataset(
            str(path), feature_names=feature_names, target_names=target_names
        )
