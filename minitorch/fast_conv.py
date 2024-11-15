from typing import Tuple
import numpy as np
from numba import njit, prange
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import Shape, Strides, broadcast_index, index_to_position, to_index
from .tensor_functions import Function

to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)

# Type aliases
Index = Tuple[int, ...]


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch, in_channels, width = input_shape
    out_channels, _, k_width = weight_shape

    for b in prange(batch):
        for oc in prange(out_channels):
            for w in prange(width):
                out_pos = index_to_position(np.array([b, oc, w]), out_strides)
                out[out_pos] = 0.0
                for ic in prange(in_channels):
                    for kw in range(k_width):
                        if reverse:
                            w_pos = w + kw
                        else:
                            w_pos = w - kw
                        if 0 <= w_pos < width:
                            in_pos = index_to_position(
                                np.array([b, ic, w_pos]), input_strides
                            )
                            w_pos = index_to_position(
                                np.array([oc, ic, kw]), weight_strides
                            )
                            out[out_pos] += input[in_pos] * weight[w_pos]


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx (Context): The context for the operation
            input (Tensor): Input tensor of shape (batch, in_channel, width)
            weight (Tensor): Weight tensor of shape (out_channel, in_channel, k_width)

        Returns:
        -------
            Tensor: Output tensor of shape (batch, out_channel, width)

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, _, k_width = weight.shape
        out = Tensor.make(
            Tensor.zeros((batch, out_channels, width)),
            (batch, out_channels, width),
            backend=input.backend,
        )
        tensor_conv1d(
            out._tensor,
            out.shape,
            out.strides,
            out._tensor.size,
            input._tensor,
            input.shape,
            input.strides,
            weight._tensor,
            weight.shape,
            weight.strides,
            False,
        )
        return out


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch, in_channels, height, width = input_shape
    out_channels, _, k_height, k_width = weight_shape

    for b in prange(batch):
        for oc in prange(out_channels):
            for h in prange(height):
                for w in prange(width):
                    out_pos = index_to_position(np.array([b, oc, h, w]), out_strides)
                    out[out_pos] = 0.0
                    for ic in prange(in_channels):
                        for kh in range(k_height):
                            for kw in range(k_width):
                                if reverse:
                                    h_pos, w_pos = h + kh, w + kw
                                else:
                                    h_pos, w_pos = h - kh, w - kw
                                if 0 <= h_pos < height and 0 <= w_pos < width:
                                    in_pos = index_to_position(
                                        np.array([b, ic, h_pos, w_pos]), input_strides
                                    )
                                    w_pos = index_to_position(
                                        np.array([oc, ic, kh, kw]), weight_strides
                                    )
                                    out[out_pos] += input[in_pos] * weight[w_pos]


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx (Context): The context for the operation
            input (Tensor): Input tensor of shape (batch, in_channel, height, width)
            weight (Tensor): Weight tensor of shape (out_channel, in_channel, k_height, k_width)

        Returns:
        -------
            Tensor: Output tensor of shape (batch, out_channel, height, width)

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, height, width = input.shape
        out_channels, _, k_height, k_width = weight.shape
        out = Tensor.make(
            Tensor.zeros((batch, out_channels, height, width)),
            (batch, out_channels, height, width),
            backend=input.backend,
        )
        tensor_conv2d(
            out._tensor,
            out.shape,
            out.strides,
            out._tensor.size,
            input._tensor,
            input.shape,
            input.strides,
            weight._tensor,
            weight.shape,
            weight.strides,
            False,
        )
        return out


conv2d = Conv2dFun.apply
