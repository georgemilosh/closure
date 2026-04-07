"""Tests for closure.module — ClosureLitModule."""

from __future__ import annotations

import pytest
import torch

from closure.models import FCNN, MLP
from closure.module import ClosureLitModule


class TestClosureLitModuleInit:
    def test_default_criterion(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net)
        assert isinstance(module.criterion, torch.nn.MSELoss)

    def test_custom_criterion(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net, criterion="L1Loss")
        assert isinstance(module.criterion, torch.nn.L1Loss)

    def test_hparams_saved(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net, lr=0.01, weight_decay=1e-4)
        assert module.hparams["lr"] == 0.01
        assert module.hparams["weight_decay"] == 1e-4


class TestClosureLitModuleForward:
    def test_training_step(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net)
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 8, 16, 16)
        loss = module.training_step((x, y), 0)
        assert loss.ndim == 0
        assert loss.requires_grad

    def test_validation_step(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net)
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 8, 16, 16)
        loss = module.validation_step((x, y), 0)
        assert loss.ndim == 0

    def test_predict_step(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net)
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 8, 16, 16)
        pred = module.predict_step((x, y), 0)
        assert pred.shape == (2, 8, 16, 16)

    def test_mlp_forward(self):
        net = MLP(feature_dims=[10, 32, 5])
        module = ClosureLitModule(network=net)
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss = module.training_step((x, y), 0)
        assert loss.ndim == 0


class TestConfigureOptimizers:
    def test_default_optimizer(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net, lr=0.005)
        result = module.configure_optimizers()
        assert "optimizer" in result
        assert isinstance(result["optimizer"], torch.optim.Adam)
        assert result["optimizer"].defaults["lr"] == 0.005

    def test_sgd_optimizer(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net, optimizer="SGD", lr=0.01)
        result = module.configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.SGD)

    def test_scheduler_returned(self):
        net = FCNN(channels=[3, 16, 8], kernels=[3, 3])
        module = ClosureLitModule(network=net, scheduler="ReduceLROnPlateau")
        result = module.configure_optimizers()
        assert "lr_scheduler" in result
        sched_cfg = result["lr_scheduler"]
        assert sched_cfg["monitor"] == "val_loss"
