"""
closure — ML framework for fluid closure of PIC plasma simulations.
"""

from closure.config import load_paths
from closure.module import ClosureLitModule
from closure.datamodule import ClosureDataModule
from closure.models import CNet, FCNN, ResNet, MLP
from closure.datasets import DataFrameDataset
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
    "get_Ohm",
    "get_PS_2D_field",
    "get_Az",
    "get_J_perp",
    "get_exp_times",
    "read_data_ipic3d",
    "build_XY",
    "__version__",
]
