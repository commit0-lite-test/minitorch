"""
Implementation of the autodifferentiation Functions for Tensor.
"""
from __future__ import annotations
import random
from typing import TYPE_CHECKING
import numpy as np
import minitorch
from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend
if TYPE_CHECKING:
    from typing import Any, List, Tuple
    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape

def wrap_tuple(x):
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)

class Function:
    @staticmethod
    def forward(ctx: Context, *args: Tensor) -> Tensor:
        raise NotImplementedError("Forward not implemented")

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError("Backward not implemented")

    @classmethod
    def apply(cls, *args):
        ctx = Context()
        result = cls.forward(ctx, *args)
        if any(arg.history is not None for arg in args):
            result.history = History(cls, ctx, args)
        return result

class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        return (d_output.f.neg_map(d_output),)

class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (d_output.f.inv_back_zip(t1, d_output),)

class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        return (d_output, d_output)

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        return (t1.f.mul_zip(d_output, t2), t1.f.mul_zip(d_output, t1))

class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        result = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        sigmoid_out = ctx.saved_values[0]
        return (d_output.f.mul_zip(sigmoid_out, sigmoid_out.f.add_zip(tensor([1.0], backend=d_output.backend), sigmoid_out.f.neg_map(sigmoid_out))),)

class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (d_output.f.relu_back_zip(t1, d_output),)

class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        t1 = ctx.saved_values[0]
        return (d_output.f.log_back_zip(t1, d_output),)

class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        result = t1.f.exp_map(t1)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        exp_out = ctx.saved_values[0]
        return (d_output.f.mul_zip(d_output, exp_out),)

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[int] = None) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, dim)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        original_shape, dim = ctx.saved_values
        if dim is None:
            return (d_output.expand(original_shape), 0.0)
        else:
            return (d_output.expand(original_shape), 0.0)

class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[int] = None) -> Tensor:
        if dim is not None:
            return a.permute(*[i for i in range(a.dims) if i != dim] + [dim]).contiguous().view(int(prod(a.shape) / a.shape[dim]), a.shape[dim]).f.mul_reduce(a)
        else:
            return a.f.mul_reduce(a.contiguous().view(a.size))

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        raise NotImplementedError("Backward not implemented for All")

class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return (zeros(a_shape), zeros(b_shape))

class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, b.shape)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        a_shape, b_shape = ctx.saved_values
        return (zeros(a_shape), zeros(b_shape))

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        return (zeros(d_output.shape), zeros(d_output.shape))

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tuple[int]) -> Tensor:
        ctx.save_for_backward(order)
        return Tensor.make(a._tensor.permute(*order), backend=a.backend)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        order = ctx.saved_values[0]
        inv_order = [0] * len(order)
        for i, j in enumerate(order):
            inv_order[j] = i
        return (Tensor.make(d_output._tensor.permute(*inv_order), backend=d_output.backend), 0.0)

class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tuple[int]) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        return Tensor.make(a._tensor._storage, shape, backend=a.backend)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, float]:
        original = ctx.saved_values[0]
        return (Tensor.make(d_output._tensor._storage, original, backend=d_output.backend), 0.0)

class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor]:
        return (d_output,)

class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, d_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        return (
            d_output.f.matrix_multiply(d_output, t2.permute(1, 0)),
            d_output.f.matrix_multiply(t1.permute(1, 0), d_output)
        )

def zeros(shape: UserShape, backend: TensorBackend=SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return Tensor.make([0] * int(prod(shape)), shape, backend=backend)

def rand(shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(prod(shape)))]
    tensor = Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def _tensor(ls: Any, shape: UserShape, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor

def tensor(ls: Any, backend: TensorBackend=SimpleBackend, requires_grad: bool=False) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    if isinstance(ls, Tensor):
        return ls
    if isinstance(ls, (float, int)):
        return _tensor([ls], (1,), backend=backend, requires_grad=requires_grad)
    if isinstance(ls, list):
        tensor = _tensor(ls, (len(ls),), backend=backend)
        tensor.requires_grad_(requires_grad)
        return tensor
    raise NotImplementedError("Couldn't create tensor from %s" % (ls,))
