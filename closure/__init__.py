"""
closure — ML framework for fluid closure of PIC plasma simulations.
"""

from closure.config import load_paths
from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule
from closure.models import CNet, FCNN, ResNet, MLP, EquilibriumAnchoredResidualPressureMLP
from closure.datasets import DataFrameDataset
from closure.run_loader import RunLoader
from closure.evaluation import transform_features
from closure.dispersion import (
    DISPERSION_PRESSURE_COMPONENTS,
    HallMHDBackground,
    MENURA_FEATURE_NAMES,
    MENURA_PRESSURE_COMPONENTS,
    apply_closure_correction,
    build_dispersion_matrix,
    build_hall_mhd_operator,
    build_menura_closure_operator,
    closure_tensor_fourier_symbol_at_equilibrium,
    closure_tensor_jacobian_at_equilibrium,
    electron_pressure_tensor_to_electric_jacobian,
    eigensystem,
    fourier_mode_vector,
    hall_mhd_k_vector,
    isotropic_electron_closure_electric_jacobian,
    linearize_spatial_model,
    linearize_spatial_model_2d,
    match_eigenbranches,
    menura_binomial_filter_transfer,
    menura_electron_velocity_jacobian,
    menura_feature_jacobian,
    menura_fourth_order_derivative_wavenumber,
    menura_fourth_order_laplacian_symbol,
    menura_pressure_primitive_jacobian,
    menura_strain_feature_jacobian,
    mode_indices_from_physical_wavenumber,
    patch_domain_lengths_from_grid,
    physical_wavenumber_from_mode_indices,
    project_fourier_jacobian,
    project_fourier_jacobian_2d,
    scan_dispersion_relation,
    operator_amplification,
)
from closure.plasma import get_Ohm, get_PS_2D_field, get_Az, get_J_perp
from closure.read_pic import get_exp_times, read_data_ipic3d, build_XY
from closure.experiments import discover_experiments, resolve_experiments

__version__ = "0.2.0"

__all__ = [
    "DISPERSION_PRESSURE_COMPONENTS",
    "load_paths",
    "ClosureLitModule",
    "ClosureDataModule",
    "CNet",
    "FCNN",
    "ResNet",
    "MLP",
    "EquilibriumAnchoredResidualPressureMLP",
    "DataFrameDataset",
    "RunLoader",
    "transform_features",
    "HallMHDBackground",
    "MENURA_FEATURE_NAMES",
    "MENURA_PRESSURE_COMPONENTS",
    "apply_closure_correction",
    "build_dispersion_matrix",
    "build_hall_mhd_operator",
    "build_menura_closure_operator",
    "closure_tensor_fourier_symbol_at_equilibrium",
    "closure_tensor_jacobian_at_equilibrium",
    "electron_pressure_tensor_to_electric_jacobian",
    "eigensystem",
    "fourier_mode_vector",
    "hall_mhd_k_vector",
    "isotropic_electron_closure_electric_jacobian",
    "linearize_spatial_model",
    "linearize_spatial_model_2d",
    "match_eigenbranches",
    "menura_binomial_filter_transfer",
    "menura_electron_velocity_jacobian",
    "menura_feature_jacobian",
    "menura_fourth_order_derivative_wavenumber",
    "menura_fourth_order_laplacian_symbol",
    "menura_pressure_primitive_jacobian",
    "menura_strain_feature_jacobian",
    "mode_indices_from_physical_wavenumber",
    "patch_domain_lengths_from_grid",
    "physical_wavenumber_from_mode_indices",
    "project_fourier_jacobian",
    "project_fourier_jacobian_2d",
    "scan_dispersion_relation",
    "operator_amplification",
    "get_Ohm",
    "get_PS_2D_field",
    "get_Az",
    "get_J_perp",
    "get_exp_times",
    "read_data_ipic3d",
    "build_XY",
    "discover_experiments",
    "resolve_experiments",
    "__version__",
]
