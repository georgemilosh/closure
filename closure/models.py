
"""
models.py — Neural network architectures for closure.

Pure ``torch.nn.Module`` implementations: CNet, ResNet, FCNN, MLP, and
InvariantFieldAlignedPressureMLP.

Repo:       closure
Projects:   STRIDE, HELIOSKILL
Author:     George Miloshevich
Date:       2025
License:    MIT License
"""

__all__ = ["CNet", "FCNN", "ResNet", "MLP", "InvariantFieldAlignedPressureMLP"]

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class CNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2) 
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5) 
        self.conv3 = nn.Conv2d(16, 32, 5) 
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.pool(F.relu(self.conv4(x))) 
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet(torch.nn.Module):
    """ResNet-style convolutional model with optional skip connections.

    Parameters
    ----------
    channels : list[int]
        Number of channels for each layer (length N+1 for N conv layers).
    kernels : list[int]
        Kernel size for each conv layer (length N).
    activations : list[str | None] or None
        Activation function names per layer (``torch.nn`` class names).
    batch_norms : list[bool | None] or None
        Whether to apply BatchNorm2d after each layer.
    dropouts : list[float | None] or None
        Dropout2d rate per layer.
    skip_connect : dict[str, int] or None
        Skip connection mapping ``{"target_layer": source_layer}``.
    """
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        activations: list[str | None] | None = None,
        batch_norms: list[bool | None] | None = None,
        dropouts: list[float | None] | None = None,
        skip_connect: dict[str, int] | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.kernels = kernels
        if activations is None:
            activations = [None] * (len(channels) - 1)
        if batch_norms is None:
            batch_norms = [None] * (len(channels) - 1)
        if dropouts is None:
            dropouts = [None] * (len(channels) - 1)
        self.skip_connect = {}
        if skip_connect is not None:
            for key, value in skip_connect.items():
                assert int(key) > value, f"Skip connection must be to higher (key) from lower layer (value), but we have {int(key) = }, {value = }"
                self.skip_connect[int(key)] = value
        self.skip_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(len(channels)-1):
            self.convs.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], padding=(kernels[i]-1)//2))
            if i in self.skip_connect:
                self.skip_convs.append(nn.Conv2d(channels[self.skip_connect[i]+1], channels[i+1], kernels[i], padding=(kernels[i]-1)//2))
            else:
                self.skip_convs.append(None)
            if activations[i] is not None:
                if isinstance(activations[i], list):
                    self.activations.append(getattr(nn, activations[i][0])(*activations[i][1:]))
                else:
                    self.activations.append(getattr(nn, activations[i])())
            else:
                self.activations.append(None)
            if batch_norms[i] is not None and batch_norms[i]:
                self.batch_norms.append(nn.BatchNorm2d(channels[i+1]))
            else:
                self.batch_norms.append(None)
            if dropouts[i] is not None and dropouts[i] > 0.0:
                self.dropouts.append(nn.Dropout2d(dropouts[i]))
            else:
                self.dropouts.append(None)

    def forward(self, x):
        """
        Forward pass of the FCNN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = x
        out_store = []
        for i in range(len(self.channels) - 1):
            out = self.convs[i](out)
            if self.skip_convs[i] is not None:
                out += self.skip_convs[i](out_store[self.skip_connect[i]])
                """try:
                    out += self.skip_convs[i](out_store[self.skip_connect[i]])
                except Exception as e:
                    logger.error(f"{i = }, {self.skip_connect[i] = }, {self.skip_convs[i] = }, {self.convs[i] = }, {out_store[self.skip_connect[i]].shape = }")
                    raise e"""
            if self.activations[i] is not None:
                out = self.activations[i](out)
            if self.batch_norms[i] is not None:
                out = self.batch_norms[i](out)
            if self.dropouts[i] is not None:
                out = self.dropouts[i](out)
            out_store.append(out)
        return out
    
class FCNN(torch.nn.Module):
    """Fully convolutional neural network.

    Parameters
    ----------
    channels : list[int]
        Number of channels for each layer (length N+1 for N conv layers).
    kernels : list[int]
        Kernel size for each conv layer (length N).
    activations : list[str | None] or None
        Activation function names per layer.
    batch_norms : list[bool | None] or None
        Whether to apply BatchNorm2d.
    dropouts : list[float | None] or None
        Dropout2d rate per layer.
    """
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        activations: list[str | None] | None = None,
        batch_norms: list[bool | None] | None = None,
        dropouts: list[float | None] | None = None,
    ):
        super().__init__()
        seq_list = []
        if activations is None:
            activations = [None] * (len(channels) - 1)
        if batch_norms is None:
            batch_norms = [None] * (len(channels) - 1)
        if dropouts is None:
            dropouts = [None] * (len(channels) - 1)
        for i in range(len(channels)-1):
            # For even kernels, need to add padding or use output_padding in upsampling
            if kernels[i] % 2 == 0:
                padding = kernels[i] // 2
            else:
                padding = (kernels[i] - 1) // 2
            seq_list.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], padding=padding))
            if activations[i] is not None:
                seq_list.append(getattr(nn, activations[i])())
            if batch_norms[i] is not None and batch_norms[i]:
                seq_list.append(nn.BatchNorm2d(channels[i+1]))
            if dropouts[i] is not None and dropouts[i] > 0.0:
                seq_list.append(nn.Dropout2d(dropouts[i]))
        self.seq_model = torch.nn.Sequential(*seq_list)

    def forward(self, x):
        """
        Forward pass of the FCNN.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        out = self.seq_model(x)
        return out
        
class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        feature_dims (list): A list of integers representing the dimensions of the input and output features.

    Attributes:
        flatten (torch.nn.Flatten): A flatten layer to convert the input tensor into a 1-dimensional tensor.
        linear_relu_stack (torch.nn.Sequential): A sequential container for the linear and ReLU layers.

    Examples:
    --------
    >>> model = MLP([8, 24, 12, 6, 1], activations=['ReLU', 'ReLU', 'ReLU', None])
    >>> print(model)
    MLP(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=8, out_features=24, bias=True)
        (1): ReLU()
        (2): Linear(in_features=24, out_features=12, bias=True)
        (3): ReLU()
        (4): Linear(in_features=12, out_features=6, bias=True)
        (5): ReLU()
        (6): Linear(in_features=6, out_features=1, bias=True)
      )
    )
    >>> model = MLP([8, 3, 1], weights=[{'name': 'uniform_', 'std' : 1/np.sqrt(8)}, {'name': 'uniform_', 'std' : 1/np.sqrt(3)}, \
        biases = [{'name': 'zeros_'},{'name': 'zeros_'}])]
    
    """
    def __init__(
        self,
        feature_dims: list[int],
        activations: list[str | None] | None = None,
        dropouts: list[float | None] | None = None,
        weights: list[dict | None] | None = None,
        biases: list[dict | None] | None = None,
    ):
        super().__init__()

        self.flatten = torch.nn.Flatten()
        seq_list = []
        if activations is None:
            activations = [None] * (len(feature_dims) - 1)
        if dropouts is None:
            dropouts = [None] * (len(feature_dims) - 1)
        if weights is None:
            weights = [None] * (len(feature_dims) - 1)
        if biases is None:
            biases = [None] * (len(feature_dims) - 1)
        for i in range(len(feature_dims) - 1):
            linear_layer = torch.nn.Linear(feature_dims[i], feature_dims[i + 1])
            if weights[i] is not None:
                w = dict(weights[i])
                name = w.pop('name')
                getattr(torch.nn.init, name)(linear_layer.weight, **w)
            if biases[i] is not None:
                b = dict(biases[i])
                name = b.pop('name')
                getattr(torch.nn.init, name)(linear_layer.bias, **b)
            seq_list.append(linear_layer)
            if activations[i] is not None:
                activation_layer = getattr(torch.nn, activations[i])() #  'ReLu' => torch.nn.ReLU() 
                seq_list.append(activation_layer)
            if dropouts is not None:
                if dropouts[i] is not None:
                    dropout_layer = torch.nn.Dropout(dropouts[i])
                    seq_list.append(dropout_layer)
        self.linear_relu_stack = torch.nn.Sequential(*seq_list)

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): The input tensor. Either pixel-wise ``[N, C]``
                (the ``flatten=true`` data path) or an image batch
                ``[B, C, H, W]`` (the ``flatten=false`` path). For image
                batches the MLP is applied per-pixel and the spatial layout is
                restored, so the output is ``[B, C_out, H, W]`` and remains
                compatible with spatially-aware losses (e.g. the physics gradP
                term, which requires 4D tensors).

        Returns:
            torch.Tensor: The output tensor, ``[N, C_out]`` for pixel-wise input
                or ``[B, C_out, H, W]`` for image input.

        """
        if x.ndim == 4:
            b, c, h, w = x.shape
            pixels = x.permute(0, 2, 3, 1).reshape(-1, c)
            out = self.linear_relu_stack(pixels)
            return out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out


class InvariantFieldAlignedPressureMLP(torch.nn.Module):
    """Equivariant pressure-tensor closure trained in a field-aligned frame.

    The learned MLP only sees four scalar invariants built from lower-order
    moments,

    ``log(|rho_e|)``, ``log(|B|)``,
    ``asinh(E_e_parallel / electric_scale)``, and
    ``asinh(|E_e_perp| / electric_scale)``,

    where ``E_e = E + u_e x B`` is the electron-frame electric field.  The
    latter is Galilean invariant under the non-relativistic transformations
    ``u_e' = u_e - U`` and ``E' = E + U x B``.  All four quantities are
    rotational scalars.

    The MLP predicts a tensor in the right-handed local basis
    ``(e_perp1, e_perp2, b)`` with the parallel component last.  A deterministic
    layer rotates it back to Cartesian coordinates, so :meth:`forward` remains
    a drop-in replacement for the standard six-channel pressure model:

    ``[Pxx, Pyy, Pzz, Pxy, Pxz, Pyz]``.

    Raw, unnormalised Alfv\'en-unit features and targets must be supplied.  The
    companion config therefore sets both dataset scalers and prescalers to
    ``false``/``null``.  ``pressure_scale`` supplies a scalar numerical scale
    internally without changing rotational covariance.

    Parameters
    ----------
    feature_dims : list[int]
        MLP dimensions.  The first and last entries must be 4 and 6.
    magnetic_indices, velocity_indices, electric_indices : list[int]
        Three input-channel indices for each vector, in Cartesian x/y/z order.
    density_index : int
        Input-channel index of electron charge density.  Its absolute value is
        used, so either charge-density sign convention is accepted.
    guide_direction : list[float]
        Fixed guide/reference direction in the input Cartesian coordinates.
        The default is ``y`` for the iPiC3D reconnection geometry.
    enforce_spd : bool
        If true, interpret the six raw outputs as a Cholesky factor and form
        ``P_field = pressure_scale * L L^T``.  This guarantees a symmetric
        positive-definite prediction.
    """

    def __init__(
        self,
        feature_dims: list[int],
        activations: list[str | None] | None = None,
        dropouts: list[float | None] | None = None,
        weights: list[dict | None] | None = None,
        biases: list[dict | None] | None = None,
        density_index: int = 0,
        magnetic_indices: list[int] | None = None,
        velocity_indices: list[int] | None = None,
        electric_indices: list[int] | None = None,
        guide_direction: list[float] | None = None,
        density_scale: float = 0.1,
        magnetic_scale: float = 1.0,
        electric_scale: float = 0.1,
        pressure_scale: float = 3.0e-3,
        frame_epsilon: float = 1.0e-8,
        cholesky_epsilon: float = 1.0e-4,
        enforce_spd: bool = True,
        frobenius_loss: bool = True,
        extra_invariant_indices: list[int] | None = None,
        extra_invariant_scales: list[float] | None = None,
        use_electron_frame_invariants: bool = True,
        strain_tensor_indices: list[int] | None = None,
        strain_frame_scale: float = 1.0,
        strain_frame_products: bool = False,
        block_loss_lambda: float = 0.0,
        block_loss_sigmas: list[float] | None = None,
    ):
        super().__init__()
        if magnetic_indices is None:
            magnetic_indices = [1, 2, 3]
        if velocity_indices is None:
            velocity_indices = [4, 5, 6]
        if electric_indices is None:
            electric_indices = [7, 8, 9]
        if guide_direction is None:
            guide_direction = [0.0, 1.0, 0.0]
        extra_invariant_indices = list(extra_invariant_indices or [])
        if extra_invariant_scales is None:
            extra_invariant_scales = [1.0] * len(extra_invariant_indices)
        if len(extra_invariant_scales) != len(extra_invariant_indices):
            raise ValueError(
                "extra_invariant_scales must match extra_invariant_indices in length"
            )
        if any(scale <= 0.0 for scale in extra_invariant_scales):
            raise ValueError("extra_invariant_scales must be positive")
        # Parity-matched inputs: the six Cartesian components of the
        # rate-of-strain tensor, rotated into the same frame as the target.
        # W_12 then shares the parity of P_12 under b -> -b, which no rotational
        # magnitude can (see the field-frame closure report).
        strain_tensor_indices = list(strain_tensor_indices or [])
        if strain_tensor_indices and len(strain_tensor_indices) != 6:
            raise ValueError(
                "strain_tensor_indices must list exactly six packed components "
                "[Wxx, Wyy, Wzz, Wxy, Wxz, Wyz]"
            )
        if strain_frame_scale <= 0.0:
            raise ValueError("strain_frame_scale must be positive")
        if strain_frame_products and not strain_tensor_indices:
            raise ValueError("strain_frame_products requires strain_tensor_indices")
        n_invariants = (2 if use_electron_frame_invariants else 0) + 2
        n_invariants += len(extra_invariant_indices)
        # six in-frame components, plus the two gyroviscous scalings W/|B| and
        # n W when requested (the physical coefficient is multiplicative)
        n_invariants += (18 if strain_frame_products else 6) if strain_tensor_indices else 0
        if len(feature_dims) < 2 or feature_dims[0] != n_invariants or feature_dims[-1] != 6:
            raise ValueError(
                f"feature_dims must start at {n_invariants} invariant inputs "
                "(log n, log |B|, optional electron-frame pair, plus any "
                "precomputed extras) and end at 6 tensor outputs"
            )
        for name, indices in (
            ("magnetic_indices", magnetic_indices),
            ("velocity_indices", velocity_indices),
            ("electric_indices", electric_indices),
        ):
            if len(indices) != 3:
                raise ValueError(f"{name} must contain exactly three x/y/z channel indices")
        if len(guide_direction) != 3:
            raise ValueError("guide_direction must have three Cartesian components")
        for name, value in (
            ("density_scale", density_scale),
            ("magnetic_scale", magnetic_scale),
            ("electric_scale", electric_scale),
            ("pressure_scale", pressure_scale),
            ("frame_epsilon", frame_epsilon),
            ("cholesky_epsilon", cholesky_epsilon),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        guide = torch.tensor(guide_direction, dtype=torch.float32)
        guide_norm = torch.linalg.vector_norm(guide)
        if float(guide_norm) == 0.0:
            raise ValueError("guide_direction must be nonzero")

        self.trunk = MLP(
            feature_dims=feature_dims,
            activations=activations,
            dropouts=dropouts,
            weights=weights,
            biases=biases,
        )
        self.density_index = int(density_index)
        self.bx_index, self.by_index, self.bz_index = (int(i) for i in magnetic_indices)
        self.vx_index, self.vy_index, self.vz_index = (int(i) for i in velocity_indices)
        self.ex_index, self.ey_index, self.ez_index = (int(i) for i in electric_indices)
        self.density_scale = float(density_scale)
        self.magnetic_scale = float(magnetic_scale)
        self.electric_scale = float(electric_scale)
        self.pressure_scale = float(pressure_scale)
        self.frame_epsilon = float(frame_epsilon)
        self.cholesky_epsilon = float(cholesky_epsilon)
        self.enforce_spd = bool(enforce_spd)
        self.frobenius_loss = bool(frobenius_loss)
        self.use_electron_frame_invariants = bool(use_electron_frame_invariants)
        # Kept as buffers, not Python lists: TorchScript cannot infer the type
        # of an empty list attribute, and the export path must stay scriptable.
        self.extra_count = len(extra_invariant_indices)
        self.register_buffer(
            "extra_invariant_index",
            torch.tensor([int(i) for i in extra_invariant_indices], dtype=torch.long),
        )
        self.register_buffer(
            "extra_invariant_scale",
            torch.tensor([float(v) for v in extra_invariant_scales], dtype=torch.float32),
        )
        self.required_channels = int(max(
            [
                self.density_index,
                self.bx_index, self.by_index, self.bz_index,
                self.vx_index, self.vy_index, self.vz_index,
                self.ex_index, self.ey_index, self.ez_index,
            ]
            + [int(i) for i in extra_invariant_indices]
            + [int(i) for i in strain_tensor_indices]
        ))
        # Block-weighted loss.  The plain Frobenius objective is rotationally
        # invariant, which is what makes it physically right -- and also what
        # stops it from emphasising the agyrotropic block, which carries ~0.05%
        # of the total variance.  Weighting the three irreducible blocks about
        # b (m = 0, +-1, +-2) is the only reweighting that keeps the loss
        # invariant: rotating about b mixes components WITHIN a block, so only
        # whole-block weights are well defined.
        if block_loss_lambda < 0.0:
            raise ValueError("block_loss_lambda must be non-negative")
        if block_loss_lambda > 0.0 and block_loss_sigmas is None:
            raise ValueError("block_loss_lambda > 0 requires block_loss_sigmas")
        if block_loss_sigmas is not None:
            if len(block_loss_sigmas) != 3:
                raise ValueError("block_loss_sigmas must be [gyrotropic, m=+-2, m=+-1]")
            if any(v <= 0.0 for v in block_loss_sigmas):
                raise ValueError("block_loss_sigmas must be positive")
            sig = torch.tensor([float(v) for v in block_loss_sigmas])
            weights = (sig[0] / sig) ** (2.0 * float(block_loss_lambda))
        else:
            weights = torch.ones(3)
        self.block_loss_lambda = float(block_loss_lambda)
        self.register_buffer("block_loss_weight", weights, persistent=False)
        self.strain_count = len(strain_tensor_indices)
        self.strain_frame_products = bool(strain_frame_products)
        self.strain_frame_scale = float(strain_frame_scale)
        self.register_buffer(
            "strain_tensor_index",
            torch.tensor([int(i) for i in strain_tensor_indices], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("guide_direction", guide / guide_norm)

    @staticmethod
    def _as_pixels(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """Return channels-last pixels plus image dimensions (zero for 2-D input)."""
        if x.ndim == 2:
            return x, 0, 0, 0
        if x.ndim == 4:
            batch, channels, height, width = x.shape
            pixels = x.permute(0, 2, 3, 1).reshape(-1, channels)
            return pixels, batch, height, width
        raise ValueError("expected features with shape [N,C] or [B,C,H,W]")

    @staticmethod
    def _restore_pixels(x: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
        if batch == 0:
            return x
        return x.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _packed_to_tensor(packed: torch.Tensor) -> torch.Tensor:
        """Unpack [Pxx,Pyy,Pzz,Pxy,Pxz,Pyz] into a symmetric 3x3 tensor."""
        if packed.ndim != 2 or packed.shape[1] != 6:
            raise ValueError("packed pressure must have shape [N,6]")
        return torch.stack(
            (
                torch.stack((packed[:, 0], packed[:, 3], packed[:, 4]), dim=1),
                torch.stack((packed[:, 3], packed[:, 1], packed[:, 5]), dim=1),
                torch.stack((packed[:, 4], packed[:, 5], packed[:, 2]), dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def _tensor_to_packed(tensor: torch.Tensor) -> torch.Tensor:
        """Pack a symmetric 3x3 tensor as [Pxx,Pyy,Pzz,Pxy,Pxz,Pyz]."""
        if tensor.ndim != 3 or tensor.shape[1] != 3 or tensor.shape[2] != 3:
            raise ValueError("pressure tensor must have shape [N,3,3]")
        return torch.stack(
            (
                tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2],
                tensor[:, 0, 1], tensor[:, 0, 2], tensor[:, 1, 2],
            ),
            dim=1,
        )

    def _vectors(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = self.required_channels
        if pixels.shape[1] <= required:
            raise ValueError(
                f"input has {pixels.shape[1]} channels but configured index {required} is required"
            )
        magnetic = torch.stack(
            (pixels[:, self.bx_index], pixels[:, self.by_index], pixels[:, self.bz_index]),
            dim=1,
        )
        velocity = torch.stack(
            (pixels[:, self.vx_index], pixels[:, self.vy_index], pixels[:, self.vz_index]),
            dim=1,
        )
        electric = torch.stack(
            (pixels[:, self.ex_index], pixels[:, self.ey_index], pixels[:, self.ez_index]),
            dim=1,
        )
        return magnetic, velocity, electric

    def _basis_and_invariants(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnetic, velocity, electric = self._vectors(pixels)
        guide = self.guide_direction.to(dtype=pixels.dtype, device=pixels.device).expand_as(magnetic)

        bnorm = torch.linalg.vector_norm(magnetic, dim=1, keepdim=True)
        valid_b = bnorm > self.frame_epsilon
        bhat = magnetic / bnorm.clamp_min(self.frame_epsilon)
        # At an exact magnetic null the parallel direction is physically
        # undefined.  Use the guide axis as a finite deterministic fallback.
        bhat = torch.where(valid_b, bhat, guide)

        guide_cross_b = torch.cross(guide, bhat, dim=1)
        cross_norm = torch.linalg.vector_norm(guide_cross_b, dim=1, keepdim=True)
        e1_regular = guide_cross_b / cross_norm.clamp_min(self.frame_epsilon)

        # When b is parallel to the guide direction, choose the Cartesian axis
        # least aligned with b and project it into the perpendicular plane.
        fallback_index = torch.argmin(torch.abs(bhat), dim=1)
        reference = F.one_hot(fallback_index, num_classes=3).to(dtype=pixels.dtype)
        e1_fallback = reference - torch.sum(reference * bhat, dim=1, keepdim=True) * bhat
        e1_fallback = F.normalize(e1_fallback, dim=1, eps=self.frame_epsilon)
        e1 = torch.where(cross_norm > self.frame_epsilon, e1_regular, e1_fallback)
        e2 = F.normalize(torch.cross(bhat, e1, dim=1), dim=1, eps=self.frame_epsilon)
        rotation = torch.stack((e1, e2, bhat), dim=1)

        electron_frame_e = electric + torch.cross(velocity, magnetic, dim=1)
        e_parallel = torch.sum(electron_frame_e * bhat, dim=1)
        e_perp_vector = electron_frame_e - e_parallel.unsqueeze(1) * bhat
        e_perp = torch.linalg.vector_norm(e_perp_vector, dim=1)
        rho_abs = torch.abs(pixels[:, self.density_index])

        channels = [
            torch.log((rho_abs / self.density_scale).clamp_min(self.frame_epsilon)),
            torch.log((bnorm[:, 0] / self.magnetic_scale).clamp_min(self.frame_epsilon)),
        ]
        if self.use_electron_frame_invariants:
            # NOTE: under Menura's Ohm's law E + u_e x B is identically
            # -(div P_e)/n + eta J - eta_h lap J, so these two channels feed the
            # closure its own output's divergence plus a hyper-resistive term
            # that has no counterpart in the kinetic training data.  Set
            # use_electron_frame_invariants=False (and supply the Tier-2
            # flow-strain extras instead) for a deployable, non-circular model.
            channels.append(torch.asinh(e_parallel / self.electric_scale))
            channels.append(torch.asinh(e_perp / self.electric_scale))
        # Precomputed rotational scalars (e.g. the flow-strain invariants of
        # closure/field_invariants.py) enter through a symmetric asinh so that
        # both signs and a wide dynamic range are handled without a prescaler.
        invariants = torch.stack(channels, dim=1)
        if self.strain_count > 0:
            packed = pixels.index_select(1, self.strain_tensor_index)
            strain_cart = self._packed_to_tensor(packed)
            # Same rotation as the output: input and target transform together,
            # so equivariance is exact and the parities match component by
            # component.
            strain_field = torch.bmm(
                rotation, torch.bmm(strain_cart, rotation.transpose(1, 2))
            )
            strain_packed = self._tensor_to_packed(strain_field)
            parts = [torch.asinh(strain_packed / self.strain_frame_scale)]
            if self.strain_frame_products:
                # The gyroviscous coefficient is multiplicative (Pi ~ n T W / |B|),
                # so hand the trunk the two products explicitly.  Both
                # multipliers are normalised by the same reference scales used
                # for log n and log |B|, so every group stays O(1) and the
                # single asinh scale conditions all three alike.
                b_ratio = self.magnetic_scale / bnorm.clamp_min(self.frame_epsilon)
                n_ratio = (rho_abs / self.density_scale).unsqueeze(1)
                parts.append(torch.asinh(strain_packed * b_ratio / self.strain_frame_scale))
                parts.append(torch.asinh(strain_packed * n_ratio / self.strain_frame_scale))
            invariants = torch.cat([invariants] + parts, dim=1)
        if self.extra_count > 0:
            extras = pixels.index_select(1, self.extra_invariant_index)
            scale = self.extra_invariant_scale.to(dtype=extras.dtype)
            invariants = torch.cat((invariants, torch.asinh(extras / scale)), dim=1)
        return rotation, invariants

    def _field_tensor(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.enforce_spd:
            return self.pressure_scale * self._packed_to_tensor(raw)

        diagonal = F.softplus(raw[:, :3]) + self.cholesky_epsilon
        zero = torch.zeros_like(diagonal[:, 0])
        lower = torch.stack(
            (
                torch.stack((diagonal[:, 0], zero, zero), dim=1),
                torch.stack((raw[:, 3], diagonal[:, 1], zero), dim=1),
                torch.stack((raw[:, 4], raw[:, 5], diagonal[:, 2]), dim=1),
            ),
            dim=1,
        )
        return self.pressure_scale * torch.bmm(lower, lower.transpose(1, 2))

    @staticmethod
    def _irreducible_blocks(packed: torch.Tensor):
        """Split a packed field-frame tensor into the m = 0, +-2, +-1 blocks.

        The three squared norms sum exactly to the Frobenius norm, so weights
        of (1, 1, 1) reproduce the plain Frobenius objective.
        """
        root2 = 2.0 ** 0.5
        p11, p22, ppar = packed[:, 0], packed[:, 1], packed[:, 2]
        p12, p1par, p2par = packed[:, 3], packed[:, 4], packed[:, 5]
        pperp = 0.5 * (p11 + p22)
        a2 = 0.5 * (p11 - p22)
        gyro = torch.stack((root2 * pperp, ppar), dim=1)
        m2 = torch.stack((root2 * a2, root2 * p12), dim=1)
        m1 = torch.stack((root2 * p1par, root2 * p2par), dim=1)
        return gyro, m2, m1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixels, batch, height, width = self._as_pixels(x)
        rotation, invariants = self._basis_and_invariants(pixels)
        field_pressure = self._field_tensor(self.trunk(invariants))
        cartesian_pressure = torch.bmm(
            rotation.transpose(1, 2),
            torch.bmm(field_pressure, rotation),
        )
        packed = self._tensor_to_packed(cartesian_pressure)
        return self._restore_pixels(packed, batch, height, width)

    @torch.jit.ignore
    def compute_training_loss(
        self,
        features: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        criterion: torch.nn.Module,
    ) -> torch.Tensor:
        """Compute loss after rotating prediction and target to the local frame.

        Off-diagonal components receive a factor ``sqrt(2)`` by default, so
        MSE on the six packed components is proportional to the tensor
        Frobenius error and is therefore invariant under orthogonal rotations.
        """
        feature_pixels, _, _, _ = self._as_pixels(features)
        prediction_pixels, _, _, _ = self._as_pixels(prediction)
        target_pixels, _, _, _ = self._as_pixels(target)
        rotation, _ = self._basis_and_invariants(feature_pixels)

        prediction_field = torch.bmm(
            rotation,
            torch.bmm(self._packed_to_tensor(prediction_pixels), rotation.transpose(1, 2)),
        )
        target_field = torch.bmm(
            rotation,
            torch.bmm(self._packed_to_tensor(target_pixels), rotation.transpose(1, 2)),
        )
        prediction_packed = self._tensor_to_packed(prediction_field) / self.pressure_scale
        target_packed = self._tensor_to_packed(target_field) / self.pressure_scale

        if self.block_loss_lambda > 0.0:
            # Weighted sum over the three irreducible blocks.  Dividing by six
            # keeps the overall scale identical to the mean-over-components
            # convention of the criterion path, so lambda = 0 is a no-op.
            w = self.block_loss_weight.to(prediction_packed.dtype)
            total = prediction_packed.new_zeros(())
            for k, (bp, bt) in enumerate(
                zip(
                    self._irreducible_blocks(prediction_packed),
                    self._irreducible_blocks(target_packed),
                )
            ):
                total = total + w[k] * ((bp - bt) ** 2).sum(dim=1).mean()
            return total / 6.0
        if self.frobenius_loss:
            weights = prediction_packed.new_tensor(
                [1.0, 1.0, 1.0, 2.0**0.5, 2.0**0.5, 2.0**0.5]
            )
            prediction_packed = prediction_packed * weights
            target_packed = target_packed * weights
        return criterion(prediction_packed, target_packed)
