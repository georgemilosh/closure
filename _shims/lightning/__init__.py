"""Shim: redirect ``import lightning`` to ``pytorch_lightning``.

This allows code written for the unified ``lightning`` package to work
with the older ``pytorch_lightning`` namespace provided by HPC modules.

Also patches ``numpy._core`` → ``numpy.core`` so that pickle files
created with NumPy 2.x can be loaded under NumPy 1.x.

Usage:
    export PYTHONPATH=/path/to/_shims:$PYTHONPATH
    module load PyTorch-Lightning/2.2.1-foss-2023a-CUDA-12.1.1
    python your_script.py
"""

import sys

# --- NumPy 2.x → 1.x pickle compatibility ---
# NumPy 2.x moved numpy.core to numpy._core.  Pickle files written with
# NumPy 2.x embed "numpy._core.*" paths that fail on NumPy 1.x.
import numpy.core
import numpy.core.multiarray
import numpy.core.numeric

sys.modules.setdefault("numpy._core", numpy.core)
sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

# --- lightning → pytorch_lightning redirect ---
import pytorch_lightning
from pytorch_lightning import *  # noqa: F403

pytorch = pytorch_lightning
sys.modules[__name__ + ".pytorch"] = pytorch_lightning
