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
