"""Shim: make ``numpy._core`` resolve to ``numpy.core`` on NumPy < 2.

NumPy 2.0 renamed ``numpy.core`` → ``numpy._core``.  Pickle files
created with NumPy 2.x embed ``numpy._core.multiarray`` etc. and
fail to unpickle under NumPy 1.x.  This shim bridges the gap.

Place this directory on PYTHONPATH so that ``import numpy._core``
falls through to the real ``numpy.core``.
"""

import numpy.core as _real_core
import sys

# Expose everything from numpy.core
from numpy.core import *  # noqa: F403

# Register submodules that pickle often references
for _submod_name in ("multiarray", "numeric", "umath", "fromnumeric",
                     "_multiarray_umath", "records", "_internal"):
    _full_old = f"numpy.core.{_submod_name}"
    _full_new = f"numpy._core.{_submod_name}"
    if _full_old in sys.modules and _full_new not in sys.modules:
        sys.modules[_full_new] = sys.modules[_full_old]
    else:
        try:
            _mod = __import__(f"numpy.core.{_submod_name}", fromlist=[_submod_name])
            sys.modules[_full_new] = _mod
        except ImportError:
            pass

# Make sure numpy._core itself is registered
sys.modules[__name__] = _real_core
