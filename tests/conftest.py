from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    path = pathlib.Path(__file__).parent / "fixtures"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def mock_simulation_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    sim_dir = tmp_path / "ipic3d_mock"
    sim_dir.mkdir()
    (sim_dir / "SimulationData.txt").write_text(
        "Simulation domain = 4 x 3 x 1\n"
        "Grid resolution = 4 x 3 x 1\n"
        "Time step size (dt) = 0.25\n"
        "Charge-to-mass ratio = -1\n"
        "Charge-to-mass ratio = 1\n"
    )

    x = np.arange(5)[:, None, None]
    y = np.arange(4)[None, :, None]
    z = np.arange(2)[None, None, :]
    base = x + 10 * y + 100 * z

    with h5py.File(sim_dir / "proc0.hdf", "w") as h5f:
        topology = h5f.create_group("topology")
        topology.create_dataset("cartesian_coord", data=np.array([0, 0, 0]))
        topology.create_dataset("cartesian_rank", data=0)

        fields = h5f.create_group("fields")
        for name, scale in {
            "Bx": 1.0,
            "By": 2.0,
            "Bz": 3.0,
            "Ex": 0.5,
            "Ey": 1.5,
            "Ez": 2.5,
            "divB": 0.1,
        }.items():
            group = fields.create_group(name)
            group.create_dataset("cycle_0", data=(base * scale + 1).astype(float))
            group.create_dataset("cycle_2", data=(base * scale + 2).astype(float))

        moments = h5f.create_group("moments")
        for species_index, sign in [(0, -1.0), (1, 1.0)]:
            species_group = moments.create_group(f"species_{species_index}")
            current_scale = 0.3 if species_index == 0 else 0.6
            for name, scale in {
                "rho": sign * 2.0,
                "Jx": sign * current_scale,
                "Jy": sign * (current_scale + 0.1),
                "Jz": sign * (current_scale + 0.2),
                "Pxx": 2.0,
                "Pxy": 0.2,
                "Pxz": 0.1,
                "Pyy": 3.0,
                "Pyz": 0.05,
                "Pzz": 4.0,
            }.items():
                group = species_group.create_group(name)
                group.create_dataset("cycle_0", data=(base * 0 + scale).astype(float))
                group.create_dataset("cycle_2", data=(base * 0 + scale + species_index).astype(float))

    return sim_dir


@pytest.fixture
def mock_hdf5_path(mock_simulation_dir: pathlib.Path) -> pathlib.Path:
    return mock_simulation_dir / "proc0.hdf"


@pytest.fixture
def mock_ecsim_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mock directory with ECSIM-style field HDF5 files."""
    sim_dir = tmp_path / "ecsim_mock"
    sim_dir.mkdir()
    (sim_dir / "SimulationData.txt").write_text(
        "Simulation domain = 4 x 3 x 1\n"
        "Grid resolution = 4 x 3 x 1\n"
        "Time step size (dt) = 0.25\n"
        "Charge-to-mass ratio = -1\n"
        "Charge-to-mass ratio = 1\n"
    )

    # ECSIM field shape: stored as (nxc+1, nyc+1, nzc+1)
    shape = (5, 4, 2)
    rng = np.random.default_rng(42)

    field_names = [
        "Bx", "By", "Bz", "Ex", "Ey", "Ez",
        "rho_0", "rho_1",
        "Jx_0", "Jy_0", "Jz_0", "Jx_1", "Jy_1", "Jz_1",
    ]

    for cycle_idx, cycle_num in enumerate([0, 2]):
        fname = f"experiment-Fields_{cycle_num:06d}.h5"
        with h5py.File(sim_dir / fname, "w") as h5f:
            block = h5f.create_group("Step#0/Block")
            for name in field_names:
                grp = block.create_group(name)
                grp.create_dataset("0", data=rng.random(shape) + cycle_idx)

    return sim_dir


# ----------------------------------------------------------------------
# Tiny on-disk NPZ fixture for LazyNPZDataFrameDataset integration tests.
# Each `.npz` is a flat dict of {fieldname: (1, H, W) float32 array}, mirroring
# the iPiC3D .npz output format (Bx, By, Bz, Ex, Ey, Ez). A minimal
# SimulationData.txt sits alongside so `parse_simulation_data` succeeds.
# ----------------------------------------------------------------------
@pytest.fixture
def tiny_npz_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a small on-disk NPZ snapshot set for lazy-loading tests.

    Layout::

        <tmp_path>/npz_run/
            SimulationData.txt
            snap_000000.npz
            snap_000001.npz
            ...
            train.csv      (filenames column listing "npz_run/snap_NNNNNN.npz")
            val.csv

    Returns the top-level tmp_path (i.e. the `data_folder` to pass to the
    dataset). CSVs reference files via the `npz_run/` subdirectory.
    """
    run_name = "npz_run"
    run_dir = tmp_path / run_name
    run_dir.mkdir()

    H, W = 8, 8
    n_train = 5
    n_val = 2
    fields = ("Bx", "By", "Bz", "Ex", "Ey", "Ez")
    rng = np.random.default_rng(20260522)

    def _write(idx: int) -> str:
        arrays = {
            name: rng.standard_normal((1, H, W)).astype(np.float32)
            for name in fields
        }
        fname = f"snap_{idx:06d}.npz"
        np.savez(run_dir / fname, **arrays)
        return f"{run_name}/{fname}"

    train_files = [_write(i) for i in range(n_train)]
    val_files = [_write(n_train + i) for i in range(n_val)]

    (run_dir / "SimulationData.txt").write_text(
        "Simulation domain = 8 x 8 x 1\n"
        "Grid resolution = 8 x 8 x 1\n"
        "Time step size (dt) = 0.125\n"
        "Charge-to-mass ratio = -1\n"
        "Charge-to-mass ratio = 1\n"
    )

    import pandas as _pd
    _pd.DataFrame({"filenames": train_files}).to_csv(tmp_path / "train.csv", index=False)
    _pd.DataFrame({"filenames": val_files}).to_csv(tmp_path / "val.csv", index=False)

    return tmp_path
