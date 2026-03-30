"""Tests for closure.visualization module — import smoke tests."""

from __future__ import annotations

import importlib


class TestVisualizationImport:
    def test_module_imports(self):
        mod = importlib.import_module("closure.visualization")
        assert hasattr(mod, "graph_pred_targets")
        assert hasattr(mod, "plot_pred_targets")

    def test_graph_pred_targets_callable(self):
        from closure.visualization import graph_pred_targets
        assert callable(graph_pred_targets)

    def test_plot_pred_targets_callable(self):
        from closure.visualization import plot_pred_targets
        assert callable(plot_pred_targets)


class TestBackwardCompat:
    def test_graph_pred_targets_via_utilities(self):
        from closure.utilities import graph_pred_targets
        assert callable(graph_pred_targets)

    def test_plot_pred_targets_via_utilities(self):
        from closure.utilities import plot_pred_targets
        assert callable(plot_pred_targets)
