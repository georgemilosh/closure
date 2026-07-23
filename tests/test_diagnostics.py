from __future__ import annotations

import numpy as np
import pandas as pd

from closure.diagnostics import (
    DEFAULT_FIELDS_TO_READ,
    FieldSpec,
    _add_notebook_recon_normalization,
    _csv_source_labels,
    _default_overlay_xlabel,
    _default_overlay_ylabel,
    _overlay_style,
    _read_flags_for_specs,
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


def test_read_flags_for_specs_scopes_to_requested_moment():
    flags = _read_flags_for_specs(parse_field_specs("rho_i"))
    assert flags is not None
    assert flags["rho"] is True
    # Everything else — crucially divB — stays off.
    assert flags["divB"] is False
    assert not any(v for k, v in flags.items() if k != "rho")


def test_read_flags_for_specs_expands_pressure_dependencies():
    # P is built inside read_data from J, rho and B (for Ppar/Pperp).
    flags = _read_flags_for_specs(parse_field_specs("Pxx_e"))
    assert flags is not None
    assert {k for k, v in flags.items() if v} == {"P", "J", "rho", "B"}


def test_read_flags_for_specs_velocity_needs_rho_and_current():
    flags = _read_flags_for_specs(parse_field_specs("Vz_e"))
    assert flags is not None
    assert {k for k, v in flags.items() if v} == {"J", "rho"}


def test_read_flags_for_specs_falls_back_for_derived_fields():
    # A derived/processed quantity we cannot map ⇒ read the full default set.
    assert _read_flags_for_specs(parse_field_specs("rho_i,beta_par")) is None
    assert _read_flags_for_specs([]) is None


def test_read_flags_for_specs_default_fields_command_scope():
    # The `fields` subcommand default must not pull in P/divB it never plots.
    flags = _read_flags_for_specs(
        parse_field_specs("Az,Ey,Ez,rho_e,rho_i,Jz_e,Jz_i,Bx,By,Bz")
    )
    assert flags is not None
    assert {k for k, v in flags.items() if v} == {"B", "E", "rho", "J"}
    assert set(flags) == set(DEFAULT_FIELDS_TO_READ)


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


def test_plot_csv_overlay_csv_select_patterns_scopes_per_file(tmp_path):
    import pytest

    menura = tmp_path / "reconnection_menura.csv"
    pd.DataFrame(
        {
            "run": ["iso_r0", "iso_r0", "iso_r1e-2", "iso_r1e-2"],
            "time": [0.0, 1.0, 0.0, 1.0],
            "recon_rate": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(menura, index=False)
    ecsim = tmp_path / "reconnection_ecsim.csv"
    pd.DataFrame(
        {
            "run": ["Le2DHGEM_RunID_5_f2", "Le2DHGEM_RunID_5_f2"],
            "time": [0.0, 1.0],
            "recon_rate": [5.0, 6.0],
        }
    ).to_csv(ecsim, index=False)

    # '*r0*' scoped to the menura CSV only -> menura filtered to iso_r0, ecsim kept whole.
    # Spy on pd.concat to capture the exact rows that reach the plot.
    from closure import diagnostics

    captured_frame = {}
    real_concat = diagnostics.pd.concat

    def spy_concat(objs, *a, **k):
        combined = real_concat(objs, *a, **k)
        captured_frame["data"] = combined
        return combined

    diagnostics.pd.concat = spy_concat
    try:
        out = plot_csv_overlay(
            [menura, ecsim],
            output=tmp_path / "overlay.png",
            x="time",
            y="recon_rate",
            csv_select_patterns={"reconnection_menura.csv": {"run": ["*r0*"]}},
        )
    finally:
        diagnostics.pd.concat = real_concat
    assert out.exists() and out.stat().st_size > 0
    kept = set(captured_frame["data"]["run"].unique())
    assert kept == {"iso_r0", "Le2DHGEM_RunID_5_f2"}  # ecsim reference survives, iso_r1e-2 dropped

    # A reference that names no given CSV is an error (typo protection).
    with pytest.raises(ValueError):
        plot_csv_overlay(
            [menura, ecsim],
            output=tmp_path / "bad.png",
            x="time",
            y="recon_rate",
            csv_select_patterns={"nope.csv": {"run": ["*"]}},
        )


def test_csv_source_labels_lengthen_only_as_far_as_needed(tmp_path):
    r0 = tmp_path / "menura" / "R0" / "profiles.csv"
    r5 = tmp_path / "menura" / "R5" / "profiles.csv"
    other_r0 = tmp_path / "ecsim" / "R0" / "profiles.csv"
    for path in (r0, r5, other_r0):
        path.parent.mkdir(parents=True)
    assert _csv_source_labels([r0, r5]) == ["R0", "R5"]

    # Same parent dir name under different subtrees: grow by one component,
    # not all the way to the absolute path - these labels go in the legend.
    assert _csv_source_labels([r0, other_r0]) == ["menura/R0", "ecsim/R0"]
    assert _csv_source_labels([r0, r5, other_r0]) == ["menura/R0", "menura/R5", "ecsim/R0"]


def test_plot_csv_overlay_separates_same_run_name_across_two_csvs(tmp_path):
    """Two batches that each ran a run of the same name must not get merged
    into one scrambled line - see _csv_source_labels' docstring."""
    r0_dir = tmp_path / "R0"
    r5_dir = tmp_path / "R5"
    r0_dir.mkdir()
    r5_dir.mkdir()
    columns = {
        "run": ["iso_GEM_1e-2_Jze.5_r0"] * 2,
        "field_label": ["P_e"] * 2,
        "projection": ["y"] * 2,
        "cut_value": [9.921875] * 2,
        "coord": [0.0, 1.0],
    }
    pd.DataFrame({**columns, "value": [1.0, 2.0]}).to_csv(r0_dir / "profiles_menura.csv", index=False)
    pd.DataFrame({**columns, "value": [10.0, 20.0]}).to_csv(r5_dir / "profiles_menura.csv", index=False)

    import matplotlib.axes

    from closure import diagnostics

    captured = {}
    real_concat = diagnostics.pd.concat

    def spy_concat(objs, *a, **k):
        combined = real_concat(objs, *a, **k)
        captured["data"] = combined
        return combined

    def _plot_and_capture_labels(**kwargs) -> list[str]:
        labels: list[str] = []
        real_plot = matplotlib.axes.Axes.plot

        def spy_plot(self, *args, **plot_kwargs):
            labels.append(plot_kwargs.get("label"))
            return real_plot(self, *args, **plot_kwargs)

        diagnostics.pd.concat = spy_concat
        matplotlib.axes.Axes.plot = spy_plot
        try:
            plot_csv_overlay(
                [r0_dir / "profiles_menura.csv", r5_dir / "profiles_menura.csv"],
                **kwargs,
            )
        finally:
            diagnostics.pd.concat = real_concat
            matplotlib.axes.Axes.plot = real_plot
        return labels

    labels = _plot_and_capture_labels(output=tmp_path / "overlay.png")
    assert captured["data"]["csv_source"].tolist() == ["R0", "R0", "R5", "R5"]
    # Two distinct series (one ax.plot call each) despite run/field_label/
    # projection/cut_value being identical across the two files.
    assert len(labels) == 2
    assert any("csv_source=R0" in label for label in labels)
    assert any("csv_source=R5" in label for label in labels)

    # csv_source is folded in even on top of an explicit group_by, since a
    # merged line-plot across two files is never a sensible result.
    labels = _plot_and_capture_labels(output=tmp_path / "overlay2.png", group_by=["run"])
    assert len(labels) == 2
    assert any("csv_source=R0" in label for label in labels)
    assert any("csv_source=R5" in label for label in labels)


def test_plot_csv_overlay_series_follow_select_order(tmp_path):
    """--run/--field list order is the plotting order, not pandas' alphabetical sort."""
    runs = ["b_run", "a_run", "c_run", "extra_run"]
    frame = pd.DataFrame(
        {
            "run": [run for run in runs for _ in range(2)],
            "time": [0.0, 1.0] * len(runs),
            "grid_frac": [1.0, 2.0] * len(runs),
        }
    )
    csv_path = tmp_path / "bands.csv"
    frame.to_csv(csv_path, index=False)

    import matplotlib.axes

    def _labels(**kwargs) -> list[str]:
        labels: list[str] = []
        real_plot = matplotlib.axes.Axes.plot

        def spy_plot(self, *args, **plot_kwargs):
            labels.append(plot_kwargs.get("label"))
            return real_plot(self, *args, **plot_kwargs)

        matplotlib.axes.Axes.plot = spy_plot
        try:
            plot_csv_overlay([csv_path], x="time", y="grid_frac", group_by=["run"], **kwargs)
        finally:
            matplotlib.axes.Axes.plot = real_plot
        return labels

    labels = _labels(
        output=tmp_path / "ordered.png",
        select={"run": ["c_run", "a_run", "b_run"]},
    )
    assert labels == ["run=c_run", "run=a_run", "run=b_run"]

    # Pattern-matched values rank by which pattern they matched; unnamed values
    # (extra_run) stay after the explicitly listed ones, alphabetical among themselves.
    labels = _labels(
        output=tmp_path / "patterned.png",
        select_patterns={"run": ["c*", "*_run"]},
    )
    assert labels == ["run=c_run", "run=a_run", "run=b_run", "run=extra_run"]

    # No explicit order -> unchanged alphabetical groupby order.
    assert _labels(output=tmp_path / "plain.png") == [
        "run=a_run",
        "run=b_run",
        "run=c_run",
        "run=extra_run",
    ]


def _derived_profile_frame() -> pd.DataFrame:
    """Two coords x {P_e, P_i, Bx, By} in the long format profiles export."""
    fields = {"P_e": [1.0, 2.0], "P_i": [3.0, 4.0], "Bx": [3.0, 5.0], "By": [4.0, 12.0]}
    return pd.DataFrame(
        {
            "run": ["R0"] * 8,
            "field": [f.split("_")[0] for f in fields for _ in range(2)],
            "species": [None] * 8,
            "field_label": [f for f in fields for _ in range(2)],
            "projection": ["y"] * 8,
            "coord": [0.0, 1.0] * len(fields),
            "value": [v for values in fields.values() for v in values],
        }
    )


def _overlay_series(csv_path, **kwargs) -> dict[str, np.ndarray]:
    """Run an overlay and capture what each series actually plotted."""
    import matplotlib.axes

    series: dict[str, np.ndarray] = {}
    real_plot = matplotlib.axes.Axes.plot

    def spy_plot(self, *args, **plot_kwargs):
        series[plot_kwargs.get("label")] = np.asarray(args[1], dtype=float)
        return real_plot(self, *args, **plot_kwargs)

    matplotlib.axes.Axes.plot = spy_plot
    try:
        plot_csv_overlay([csv_path], **kwargs)
    finally:
        matplotlib.axes.Axes.plot = real_plot
    return series


def test_plot_csv_overlay_derived_field_combines_labels(tmp_path):
    csv_path = tmp_path / "profiles.csv"
    _derived_profile_frame().to_csv(csv_path, index=False)

    # B is not a field_label: it resolves to the magnitude of the components.
    expression = "P_e+P_i+B^2/(8*pi)"
    series = _overlay_series(
        csv_path,
        output=tmp_path / "derived.png",
        group_by=["field_label"],
        select={"field_label": [expression]},
        derived={expression: expression},
    )
    assert list(series) == [f"field_label={expression}"]
    np.testing.assert_allclose(
        series[f"field_label={expression}"],
        [1.0 + 3.0 + 25.0 / (8 * np.pi), 2.0 + 4.0 + 169.0 / (8 * np.pi)],
    )

    # '^' is exponentiation, not XOR: as XOR it binds looser than '/' and the
    # expression above would silently evaluate as B**(2/(8*pi)).
    powers = _overlay_series(
        csv_path,
        output=tmp_path / "power.png",
        group_by=["field_label"],
        select={"field_label": ["Bx^2", "Bx**2"]},
        derived={"Bx^2": "Bx^2", "Bx**2": "Bx**2"},
    )
    np.testing.assert_allclose(powers["field_label=Bx^2"], [9.0, 25.0])
    np.testing.assert_allclose(powers["field_label=Bx**2"], [9.0, 25.0])


def test_plot_csv_overlay_derived_field_plots_beside_plain_fields(tmp_path):
    """A derived label filters, orders and groups like any exported field."""
    csv_path = tmp_path / "profiles.csv"
    _derived_profile_frame().to_csv(csv_path, index=False)

    series = _overlay_series(
        csv_path,
        output=tmp_path / "mixed.png",
        group_by=["field_label"],
        select={"field_label": ["P_tot", "P_e"]},
        derived={"P_tot": "P_e+P_i"},
    )
    assert list(series) == ["field_label=P_tot", "field_label=P_e"]
    np.testing.assert_allclose(series["field_label=P_tot"], [4.0, 6.0])
    np.testing.assert_allclose(series["field_label=P_e"], [1.0, 2.0])


def test_plot_csv_overlay_derived_field_rejects_unsafe_expressions(tmp_path):
    import pytest

    csv_path = tmp_path / "profiles.csv"
    _derived_profile_frame().to_csv(csv_path, index=False)

    def _run(expression: str):
        plot_csv_overlay(
            [csv_path],
            output=tmp_path / "bad.png",
            select={"field_label": [expression]},
            derived={expression: expression},
        )

    with pytest.raises(ValueError, match="Unsupported function"):
        _run("__import__('os').system('true')")
    with pytest.raises(ValueError, match="Unsupported syntax"):
        _run("P_e if P_i else P_i")
    with pytest.raises(KeyError, match="neither an available field_label"):
        _run("P_e+P_missing")


def test_cli_overlay_parses_expression_fields(monkeypatch):
    from closure import diagnostics_cli

    captured = {}

    def fake_overlay(csvs, **kwargs):
        captured.update(kwargs)
        return "overlay.png"

    monkeypatch.setattr(diagnostics_cli, "plot_csv_overlay", fake_overlay)

    args = diagnostics_cli.build_parser().parse_args(
        [
            "overlay",
            "diagnostics/profiles_menura.csv",
            # bare name, named expression, bare expression - and a comma inside
            # a call must not split the entry.
            "--field",
            "P_e, P_tot=P_e+P_i, B^2/(8*pi), maximum(P_e,P_i)",
        ]
    )
    args.func(args)

    assert captured["select"]["field_label"] == [
        "P_e",
        "P_tot",
        "B^2/(8*pi)",
        "maximum(P_e,P_i)",
    ]
    assert captured["derived"] == {
        "P_tot": "P_e+P_i",
        "B^2/(8*pi)": "B^2/(8*pi)",
        "maximum(P_e,P_i)": "maximum(P_e,P_i)",
    }


def test_cli_overlay_csv_run_pattern_builds_scoped_selection(monkeypatch):
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
            "diagnostics/reconnection_menura.csv",
            "diagnostics/reconnection_ecsim.csv",
            "--csv-run-pattern",
            "reconnection_menura.csv",
            "*r0*, *r1e-4*",
        ]
    )
    args.func(args)

    assert captured["csv_select_patterns"] == {
        "reconnection_menura.csv": {"run": ["*r0*", "*r1e-4*"]}
    }


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


def test_cmd_reconnection_isolates_failing_experiment(tmp_path, monkeypatch):
    from closure import diagnostics_cli

    output = tmp_path / "reconnection.csv"

    def fake_one(_args, exp):
        if exp == "R7/bad_run":
            raise ValueError("Shape of array too small to calculate a numerical gradient")
        return pd.DataFrame({"run": [exp], "time": [0.0], "recon_rate": [1.0]})

    monkeypatch.setattr(diagnostics_cli, "_reconnection_one_experiment", fake_one)

    args = diagnostics_cli.build_parser().parse_args(
        [
            "reconnection",
            "R7/bad_run",
            "R7/good_run",
            "--backend",
            "menura",
            "--output-csv",
            str(output),
            "--csv-mode",
            "replace",
        ]
    )
    # Serial path (experiment_workers defaults to 1): the bad run is skipped, not fatal.
    args.func(args)
    written = pd.read_csv(output)
    assert list(written["run"]) == ["R7/good_run"]


def test_cmd_reconnection_raises_when_all_experiments_fail(tmp_path, monkeypatch):
    import pytest

    from closure import diagnostics_cli

    output = tmp_path / "reconnection.csv"

    def fake_all_fail(_args, exp):
        raise ValueError("boom")

    monkeypatch.setattr(diagnostics_cli, "_reconnection_one_experiment", fake_all_fail)

    args = diagnostics_cli.build_parser().parse_args(
        ["reconnection", "R7/a", "R7/b", "--output-csv", str(output)]
    )
    with pytest.raises(SystemExit):
        args.func(args)
    assert not output.exists()


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