"""
closure — ML framework for fluid closure of PIC plasma simulations.
"""

from closure.config import load_paths
from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule
from closure.models import CNet, FCNN, ResNet, MLP
from closure.datasets import DataFrameDataset
from closure.run_loader import RunLoader
from closure.evaluation import transform_features
from closure.dispersion import (
    HallMHDBackground,
    apply_closure_correction,
    build_dispersion_matrix,
    build_hall_mhd_operator,
    closure_tensor_jacobian_at_equilibrium,
    electron_pressure_tensor_to_electric_jacobian,
    eigensystem,
    fourier_mode_vector,
    isotropic_electron_closure_electric_jacobian,
    linearize_spatial_model,
    linearize_spatial_model_2d,
    match_eigenbranches,
    project_fourier_jacobian,
    project_fourier_jacobian_2d,
    scan_dispersion_relation,
)
from closure.plasma import get_Ohm, get_PS_2D_field, get_Az, get_J_perp
from closure.read_pic import get_exp_times, read_data_ipic3d, build_XY

__version__ = "0.2.0"

__all__ = [
    "load_paths",
    "ClosureLitModule",
    "ClosureDataModule",
    "CNet",
    "FCNN",
    "ResNet",
    "MLP",
    "DataFrameDataset",
    "RunLoader",
    "transform_features",
    "HallMHDBackground",
    "apply_closure_correction",
    "build_dispersion_matrix",
    "build_hall_mhd_operator",
    "closure_tensor_jacobian_at_equilibrium",
    "electron_pressure_tensor_to_electric_jacobian",
    "eigensystem",
    "fourier_mode_vector",
    "isotropic_electron_closure_electric_jacobian",
    "linearize_spatial_model",
    "linearize_spatial_model_2d",
    "match_eigenbranches",
    "project_fourier_jacobian",
    "project_fourier_jacobian_2d",
    "scan_dispersion_relation",
    "get_Ohm",
    "get_PS_2D_field",
    "get_Az",
    "get_J_perp",
    "get_exp_times",
    "read_data_ipic3d",
    "build_XY",
    "__version__",
]
