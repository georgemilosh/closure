"""VTK-focused tests for closure.read_pic."""

from __future__ import annotations

import importlib

import numpy as np

from closure import read_pic as rp


def _write_legacy_vtk_vector(path, array_name, data_zyx3):
    """Write a legacy VTK STRUCTURED_POINTS vector file (binary float32)."""
    nz, ny, nx, ncomp = data_zyx3.shape
    assert ncomp == 3
    npoints = nx * ny * nz

    header = (
        "# vtk DataFile Version 2.0\n"
        "Synthetic vector field\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        "ORIGIN 0 0 0\n"
        "SPACING 1 1 1\n"
        f"POINT_DATA {npoints}\n"
        f"VECTORS {array_name} float\n"
    )

    points = np.asarray(data_zyx3, dtype=np.float32).reshape(npoints, 3, order="C")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(points.astype(">f4", copy=False).tobytes())


def _write_legacy_vtk_scalar(path, array_name, data_zyx):
    """Write a legacy VTK STRUCTURED_POINTS scalar file (binary float32)."""
    nz, ny, nx = data_zyx.shape
    npoints = nx * ny * nz

    header = (
        "# vtk DataFile Version 2.0\n"
        "Synthetic scalar field\n"
        "BINARY\n"
        "DATASET STRUCTURED_POINTS\n"
        f"DIMENSIONS {nx} {ny} {nz}\n"
        "ORIGIN 0 0 0\n"
        "SPACING 1 1 1\n"
        f"POINT_DATA {npoints}\n"
        f"SCALARS {array_name} float\n"
        "LOOKUP_TABLE default\n"
    )

    values = np.asarray(data_zyx, dtype=np.float32).reshape(npoints, order="C")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(values.astype(">f4", copy=False).tobytes())


def _make_vtk_run(tmp_path):
    """Create a minimal VTK-based run directory for tests."""
    run_dir = tmp_path / "vtk_run"
    run_dir.mkdir()

    (run_dir / "SimulationData.txt").write_text(
        "Simulation domain = 4 x 3 x 1\n"
        "Grid resolution = 4 x 3 x 1\n"
        "Time step size (dt) = 0.25\n"
        "Charge-to-mass ratio = -1\n"
        "Charge-to-mass ratio = 1\n"
    )

    nx, ny, nz = 4, 3, 1
    bx = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    rho0 = np.zeros((nz, ny, nx), dtype=np.float32)
    j0 = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    pxx0 = np.zeros((nz, ny, nx), dtype=np.float32)

    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                base = x + 10 * y + 100 * z
                bx[z, y, x, :] = [base + 0.1, base + 0.2, base + 0.3]
                rho0[z, y, x] = 1000.0 + base
                j0[z, y, x, :] = [base + 1.0, base + 2.0, base + 3.0]
                pxx0[z, y, x] = 2000.0 + base

    # Two cycles for cycle/time extraction tests.
    _write_legacy_vtk_vector(run_dir / "GEMHarris_B_400.vtk", "B", bx)
    _write_legacy_vtk_vector(run_dir / "GEMHarris_B_600.vtk", "B", bx + 10.0)

    _write_legacy_vtk_scalar(run_dir / "GEMHarris_rho0_400.vtk", "rho0", rho0)
    _write_legacy_vtk_vector(run_dir / "GEMHarris_J0_400.vtk", "J0", j0)
    _write_legacy_vtk_scalar(run_dir / "GEMHarris_PXX0_400.vtk", "PXX0", pxx0)

    return run_dir


def test_parse_vtk_filename_and_field_mapping():
    assert rp._parse_vtk_filename("GEMHarris_B_400.vtk") == ("GEMHarris", "B", 400)
    assert rp._parse_vtk_filename("not_a_vtk_name") is None

    assert rp._fieldname_to_vtk_token("Bx") == ("B", 0)
    assert rp._fieldname_to_vtk_token("Jy_3") == ("J3", 1)
    assert rp._fieldname_to_vtk_token("Pxy_2") == ("PXY2", None)
    assert rp._fieldname_to_vtk_token("rho_1") == ("rho1", None)


def test_read_fieldname_reads_vtk_vector_and_scalar_with_remap(tmp_path):
    run_dir = _make_vtk_run(tmp_path)

    bx = rp.read_fieldname(
        str(run_dir),
        "GEMHarris_B_400.vtk",
        "Bx",
        choose_x=[0, 4],
        choose_y=[0, 3],
        choose_z=[0, 1],
    )
    rho0 = rp.read_fieldname(
        str(run_dir),
        "GEMHarris_B_400.vtk",  # intentionally pass B file: should remap to rho0 file
        "rho_0",
        choose_x=[0, 4],
        choose_y=[0, 3],
        choose_z=[0, 1],
    )

    assert bx.shape == (4, 3, 1)
    assert rho0.shape == (4, 3, 1)

    assert bx.dtype == np.dtype(">f4")
    assert rho0.dtype == np.dtype(">f4")

    # Output indexing is (x, y, t) for this 2D case.
    assert np.isclose(float(bx[2, 1, 0]), 12.1)
    assert np.isclose(float(rho0[3, 2, 0]), 1023.0)


def test_vtk_reader_fallback_without_vtk_python(monkeypatch, tmp_path):
    run_dir = _make_vtk_run(tmp_path)

    original_import_module = importlib.import_module

    def _fake_import_module(name, package=None):
        if name.startswith("vtk"):
            raise ImportError("forced missing vtk")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    arr = rp._read_vtk_field(str(run_dir / "GEMHarris_B_400.vtk"), "Bx", "B", 0)
    assert arr.shape == (1, 3, 4)
    assert np.isclose(float(arr[0, 1, 2]), 12.1)


def test_vtk_run_info_and_cycles(tmp_path):
    run_dir = _make_vtk_run(tmp_path)

    names = rp._collect_experiment_filenames(str(run_dir))
    assert names[:2] == ["GEMHarris_B_400.vtk", "GEMHarris_B_600.vtk"]

    info = rp.ecsim_available_run_info(str(run_dir))
    assert info["cycles"] == [400, 600]
    assert set(["Bx", "By", "Bz"]).issubset(set(info["fields"]))
    assert "rho" in info["fields"]
    assert "Jx" in info["fields"]
    assert "Pxx" in info["fields"]
    assert info["species_indices"] == [0]
    assert info["qom"] == [-1.0, 1.0]


def test_get_saved_iterations_supports_vtk(tmp_path):
    base_dir = tmp_path / "base"
    exp_name = "exp1"
    run_dir = base_dir / exp_name
    run_dir.mkdir(parents=True)

    (run_dir / "SimulationData.txt").write_text(
        "Simulation domain = 4 x 3 x 1\n"
        "Grid resolution = 4 x 3 x 1\n"
        "Time step size (dt) = 0.5\n"
        "Charge-to-mass ratio = -1\n"
    )

    data = np.zeros((1, 2, 2, 3), dtype=np.float32)
    _write_legacy_vtk_vector(run_dir / "GEMHarris_B_100.vtk", "B", data)
    _write_legacy_vtk_vector(run_dir / "GEMHarris_B_300.vtk", "B", data)

    iterations, times = rp.get_saved_iterations(str(base_dir), exp_name)
    assert iterations == [100, 300]
    assert times == [50.0, 150.0]
