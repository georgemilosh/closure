from __future__ import annotations

import numpy as np
import pandas as pd

from closure.diagnostics import (
    FieldSpec,
    apply_normalization,
    build_profiles_dataframe,
    discover_menura_iterations,
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


def test_apply_normalization_sample_uses_notebook_values(monkeypatch):
    X, Y = _toy_grid()
    data = _toy_data()
    calls = {}

    def fake_code2alfven(data_arg, x, y, times, b0x=None, nb=None, experiment=None):
        calls.update({"b0x": b0x, "nb": nb, "experiment": experiment, "times": times})
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