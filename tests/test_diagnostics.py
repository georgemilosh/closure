from __future__ import annotations

import numpy as np
import pandas as pd

from closure.diagnostics import (
    FieldSpec,
    _add_notebook_recon_normalization,
    _default_overlay_xlabel,
    _default_overlay_ylabel,
    _overlay_style,
    apply_normalization,
    build_profiles_dataframe,
    discover_menura_iterations,
    discover_menura_runs,
    parse_field_specs,
    plot_csv_overlay,
    plot_field_panels,
    resolve_field_data,
)


def _toy_grid():
    x = np.arange(4.0)
    y = np.arange(3.0) * 10.0
    return np.meshgrid(x, y, indexing="ij")


def _toy_data():
    base = np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2)
    return {
        "Bx": base,
        "P": {"e": base + 100.0, "i": base + 200.0},
        "rho": {"e": np.ones_like(base), "i": 2.0 * np.ones_like(base)},
    }


def test_parse_field_specs_accepts_aliases_and_species_suffixes():
    specs = parse_field_specs("Jztot,P_e,beta_par_minus_beta_perp")
    assert specs == [
        FieldSpec("Jz-tot"),
        FieldSpec("P", "e"),
        FieldSpec("beta_par - beta_perp"),
    ]


def test_resolve_field_data_defaults_to_e_for_species_dict():
    field_data, species = resolve_field_data(_toy_data(), FieldSpec("P"))
    assert species == "e"
    assert field_data.shape == (4, 3, 2)


def test_build_profiles_dataframe_exports_y_cut_at_x_index():
    X, Y = _toy_grid()
    frame = build_profiles_dataframe(
        _toy_data(),
        X,
        Y,
        [FieldSpec("P", "i")],
        run_name="R5",
        times=[0.0, 0.5],
        time_indices=[1],
        projection="y",
        cut_index=2,
    )
    assert list(frame["coord"]) == [0.0, 10.0, 20.0]
    assert frame["run"].unique().tolist() == ["R5"]
    assert frame["field_label"].unique().tolist() == ["P_i"]
    assert frame["cut_axis"].unique().tolist() == ["x"]
    assert frame["cut_value"].unique().tolist() == [2.0]
    np.testing.assert_allclose(frame["value"], _toy_data()["P"]["i"][2, :, 1])


def test_plot_field_panels_writes_png(tmp_path):
    X, Y = _toy_grid()
    output = tmp_path / "fields.png"
    result = plot_field_panels(
        _toy_data(),
        X,
        Y,
        [FieldSpec("Bx"), FieldSpec("rho", "e")],
        run_name="R0",
        output=output,
    )
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_csv_overlay_writes_png(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    pd.DataFrame(
        {
            "diagnostic": ["profile", "profile", "profile", "profile"],
            "run": ["R0", "R0", "R1", "R1"],
            "field_label": ["Bx", "Bx", "Bx", "Bx"],
            "projection": ["y", "y", "y", "y"],
            "cut_value": [2.0, 2.0, 2.0, 2.0],
            "coord": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 1.5, 2.5],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "overlay.png"
    result = plot_csv_overlay([csv_path], output=output)
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0

    log_output = tmp_path / "overlay_log.png"
    log_result = plot_csv_overlay([csv_path], output=log_output, logx=True, logy=True)
    assert log_result == log_output
    assert log_output.exists()
    assert log_output.stat().st_size > 0


def test_plot_csv_overlay_select_filters_field(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    pd.DataFrame(
        {
            "run": ["R0", "R0", "R0", "R0"],
            "field_label": ["Bx", "Bx", "P_e", "P_e"],
            "coord": [0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "p_e.png"
    result = plot_csv_overlay([csv_path], output=output, select={"field_label": ["P_e"]})
    assert result == output and output.exists() and output.stat().st_size > 0

    import pytest

    with pytest.raises(ValueError):
        plot_csv_overlay([csv_path], output=tmp_path / "none.png", select={"field_label": ["missing"]})
    with pytest.raises(KeyError):
        plot_csv_overlay([csv_path], output=tmp_path / "bad.png", select={"nope": ["x"]})


def test_plot_csv_overlay_select_patterns_filters_runs(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    pd.DataFrame(
        {
            "run": ["R0_f2", "R0_f2", "R1_f2", "R1_f2", "R0_f4", "R0_f4"],
            "field_label": ["Bx"] * 6,
            "coord": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "value": [1.0, 2.0, 1.5, 2.5, 3.0, 4.0],
        }
    ).to_csv(csv_path, index=False)
    output = tmp_path / "f2.png"
    result = plot_csv_overlay([csv_path], output=output, select_patterns={"run": ["*_f2"]})
    assert result == output and output.exists() and output.stat().st_size > 0

    import pytest

    with pytest.raises(ValueError):
        plot_csv_overlay([csv_path], output=tmp_path / "none.png", select_patterns={"run": ["nomatch*"]})
    with pytest.raises(KeyError):
        plot_csv_overlay([csv_path], output=tmp_path / "bad.png", select_patterns={"nope": ["*"]})


def test_overlay_default_labels():
    prof = pd.DataFrame({"projection": ["y", "y"], "field_label": ["P_e", "P_e"]})
    assert _default_overlay_xlabel("coord", prof) == "y"
    assert _default_overlay_ylabel("value", prof) == "P_e"
    # mixed fields -> generic; reconnection columns -> friendly label
    mixed = pd.DataFrame({"field_label": ["P_e", "Bx"]})
    assert _default_overlay_ylabel("value", mixed) == "value"
    assert _default_overlay_xlabel("time_norm", mixed) == "time"
    assert _default_overlay_ylabel("recon_rate_norm", mixed) == "reconnection rate"


def test_overlay_style_ramps_width_and_alpha():
    first = _overlay_style(0, 3)
    last = _overlay_style(2, 3)
    assert first["linewidth"] > last["linewidth"]  # width ramps down
    assert first["alpha"] < last["alpha"]  # alpha ramps up
    assert first["color"] != last["color"]
    # single series uses the max width / min alpha endpoints, no div-by-zero
    solo = _overlay_style(0, 1)
    assert solo["linewidth"] == 5.0 and solo["alpha"] == 0.35


def test_notebook_recon_normalization_matches_cell6():
    # After alfven-sample normalization Bx[0,0,0] == -1; use a normalized-like dict.
    frame = pd.DataFrame({"time": [0.0, 1.0, 2.0], "recon_rate": [1.0, -2.0, 3.0]})
    data = {
        "Bx": np.full((2, 2, 1), -1.0),
        "rho": {"e": np.full((2, 2, 1), -0.25)},
    }
    _add_notebook_recon_normalization(frame, data)
    scale = np.sqrt(0.25 * 4.0 * np.pi) / (-1.0) ** 2
    np.testing.assert_allclose(frame["time_norm"], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(frame["recon_rate_norm"], -np.array([1.0, -2.0, 3.0]) * scale)


def test_notebook_recon_normalization_scales_by_b0x():
    # Unnormalized data: time scales by |Bx0| and rate by sqrt(4*pi*rho)/Bx0**2.
    frame = pd.DataFrame({"time": [0.0, 1.0], "recon_rate": [1.0, 1.0]})
    data = {
        "Bx": np.full((2, 2, 1), -2.0),
        "rho": {"e": np.full((2, 2, 1), -1.0)},
    }
    _add_notebook_recon_normalization(frame, data)
    scale = np.sqrt(1.0 * 4.0 * np.pi) / (-2.0) ** 2
    np.testing.assert_allclose(frame["time_norm"], [0.0, 2.0])
    np.testing.assert_allclose(frame["recon_rate_norm"], [-scale, -scale])


def test_apply_normalization_sample_uses_notebook_values(monkeypatch):
    X, Y = _toy_grid()
    data = _toy_data()
    calls = {}

    def fake_code2alfven(data_arg, x, y, times, b0x=None, nb=None, experiment=None, normalize_density=True):
        calls.update({"b0x": b0x, "nb": nb, "experiment": experiment, "times": times, "normalize_density": normalize_density})
        return x + 1, y + 1, [t + 1 for t in times]

    monkeypatch.setattr("closure.diagnostics.plasma.code2alfven", fake_code2alfven)
    Xn, Yn, tn = apply_normalization(
        data,
        X,
        Y,
        [0.0, 0.5],
        normalization="alfven-sample",
        nb_factor=4 * np.pi,
    )
    assert calls["b0x"] == -data["Bx"][0, 0, 0]
    assert calls["nb"] == 4 * np.pi * np.nanmax(data["rho"]["i"][..., 0])
    assert calls["experiment"] is None
    np.testing.assert_allclose(Xn, X + 1)
    np.testing.assert_allclose(Yn, Y + 1)
    assert tn == [1.0, 1.5]


def test_cli_overlay_run_and_pattern_build_selections(monkeypatch):
    from closure import diagnostics_cli

    captured = {}

    def fake_overlay(csvs, **kwargs):
        captured["csvs"] = csvs
        captured.update(kwargs)
        return "overlay.png"

    monkeypatch.setattr(diagnostics_cli, "plot_csv_overlay", fake_overlay)

    args = diagnostics_cli.build_parser().parse_args(
        [
            "overlay",
            "diagnostics/profiles.csv",
            "--run",
            "R0_f2, R1_f2",
            "--field",
            "Bx",
            "--run-pattern",
            "R*_f4",
            "--select-pattern",
            "field_label=B*",
        ]
    )
    args.func(args)

    assert captured["select"] == {"field_label": ["Bx"], "run": ["R0_f2", "R1_f2"]}
    assert captured["select_patterns"] == {"run": ["R*_f4"], "field_label": ["B*"]}


def test_cli_accepts_menura_backend_and_normalization_options():
    from closure.diagnostics_cli import build_parser

    args = build_parser().parse_args(
        [
            "fields",
            "R0/iso_GEM",
            "--backend",
            "menura",
            "--files-path",
            "/tmp/menura/runs",
            "--normalization",
            "alfven-sample",
            "--sample-nb-factor",
            "4pi",
            "--menura-scale-ranges",
            "--choose-x",
            "0,512",
        ]
    )
    assert args.backend == "menura"
    assert args.normalization == "alfven-sample"
    assert args.sample_nb_factor == 4 * np.pi
    assert args.menura_scale_ranges is True


def test_write_csv_appends_without_repeating_header(tmp_path):
    from closure.diagnostics_cli import _write_csv

    output = tmp_path / "reconnection.csv"
    first = pd.DataFrame({"run": ["R0"], "time": [0.0], "recon_rate": [1.0]})
    second = pd.DataFrame({"run": ["R5"], "time": [1.0], "recon_rate": [2.0]})

    action, previous_rows, new_rows = _write_csv(first, output, mode="append")
    assert (action, previous_rows, new_rows) == ("created", 0, 1)
    action, previous_rows, new_rows = _write_csv(second, output, mode="append")
    assert (action, previous_rows, new_rows) == ("appended", 1, 1)

    lines = output.read_text().strip().splitlines()
    assert lines[0] == "run,time,recon_rate"
    assert len(lines) == 3


def test_write_csv_rejects_append_schema_mismatch(tmp_path):
    from closure.diagnostics_cli import _write_csv

    output = tmp_path / "reconnection.csv"
    _write_csv(pd.DataFrame({"run": ["R0"], "time": [0.0]}), output, mode="append")

    with np.testing.assert_raises(ValueError):
        _write_csv(pd.DataFrame({"run": ["R5"], "recon_rate": [1.0]}), output, mode="append")


def test_discover_menura_iterations_uses_nested_experiment_folder(tmp_path):
    products = tmp_path / "R0" / "iso_GEM" / "products"
    products.mkdir(parents=True)
    for iteration in [4000, 0, 8000]:
        (products / f"B_it{iteration}_rank_0_0.npy").touch()

    assert discover_menura_iterations(tmp_path, "R0/iso_GEM") == [0, 4000, 8000]


def _make_menura_run(run_dir):
    products = run_dir / "products"
    products.mkdir(parents=True)
    for iteration in [0, 100]:
        (products / f"B_it{iteration}_rank_0_0.npy").touch()


def test_discover_menura_runs_expands_parent_folder(tmp_path):
    for name in ["new_FCNN_00172", "old_MLP_00268", "new_MLP_00574"]:
        _make_menura_run(tmp_path / "R5" / name)
    # A non-run folder beneath R5 should be ignored.
    (tmp_path / "R5" / "logs").mkdir()

    assert discover_menura_runs(tmp_path, "R5") == [
        "R5/new_FCNN_00172",
        "R5/new_MLP_00574",
        "R5/old_MLP_00268",
    ]


def test_discover_menura_runs_keeps_single_run_unchanged(tmp_path):
    _make_menura_run(tmp_path / "R5" / "new_FCNN_00172")

    assert discover_menura_runs(tmp_path, "R5/new_FCNN_00172") == ["R5/new_FCNN_00172"]


def test_discover_menura_runs_handles_run_prefix_wrapper(tmp_path):
    _make_menura_run(tmp_path / "R5" / "run_iso_GEM")

    assert discover_menura_runs(tmp_path, "R5") == ["R5/iso_GEM"]


def test_discover_menura_runs_missing_path_passthrough(tmp_path):
    assert discover_menura_runs(tmp_path, "R5/does_not_exist") == ["R5/does_not_exist"]