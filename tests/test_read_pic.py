"""Tests for closure.read_pic."""

from __future__ import annotations

import numpy as np

from closure import read_pic as rp


class TestSimulationParsing:
    def test_parse_simulation_data_new_format(self, mock_simulation_dir):
        parsed = rp.parse_simulation_data(mock_simulation_dir)
        assert parsed["Lx"] == 4.0
        assert parsed["Ly"] == 3.0
        assert parsed["nxc"] == 4
        assert parsed["nyc"] == 3
        assert parsed["qom"] == [-1.0, 1.0]


class TestCoordinates:
    def test_build_xy_returns_expected_shape(self, mock_simulation_dir):
        x, y = rp.build_XY(mock_simulation_dir)
        assert x.shape == (4, 3)
        assert y.shape == (4, 3)


class TestIpic3dIO:
    def test_available_cycles(self, mock_simulation_dir):
        cycles, times = rp.ipic3D_available_cycles(mock_simulation_dir)
        assert cycles == [0, 2]
        assert times == [0.0, 0.5]

    def test_read_data_ipic3d(self, mock_simulation_dir):
        fields_to_read = {"B": True, "E": True, "rho": True, "J": True, "P": True, "PI": False, "divB": False, "B_ext": False, "E_ext": False, "N": False, "Qrem": False, "Heat_flux": False, "EF": False}
        data = rp.read_data_ipic3d(
            mock_simulation_dir,
            cycles=[0, 2],
            fields_to_read=fields_to_read,
            choose_species=["e", "i"],
        )
        assert data["Bx"].shape == (4, 3, 2)
        assert set(data["rho"].keys()) == {"e", "i"}
        assert data["Pxx"]["e"].shape == (4, 3, 2)

    def test_get_exp_times(self, mock_ecsim_dir):
        fields_to_read = {"B": True, "E": False, "rho": False, "J": False, "P": False, "PI": False, "divB": False, "B_ext": False, "E_ext": False, "N": False, "Qrem": False, "Heat_flux": False, "EF": False}
        data, x, y, qom, times = rp.get_exp_times([mock_ecsim_dir.name], str(mock_ecsim_dir.parent) + "/", fields_to_read, choose_species=["e", "i"])
        assert mock_ecsim_dir.name in data
        assert x.shape == (4, 3)
        assert y.shape == (4, 3)
        assert qom == [-1.0, 1.0]
        np.testing.assert_allclose(times, [0.0, 0.5])
