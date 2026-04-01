"""
closure — ML framework for fluid closure of PIC plasma simulations.
"""

from closure.config import TrainerConfig, load_paths, load_config
from closure.trainers import Trainer
from closure.models import PyNet
from closure.datasets import DataFrameDataset
from closure.plasma import get_Ohm, get_PS_2D_field, get_Az, get_J_perp
from closure.read_pic import get_exp_times, read_data_ipic3d, build_XY

__version__ = "0.1.0"

__all__ = [
    "TrainerConfig",
    "load_paths",
    "load_config",
    "Trainer",
    "PyNet",
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
