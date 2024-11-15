from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Type
import numpy as np
from typing_extensions import Protocol
from . import operators
from .tensor_data import broadcast_index, index_to_position, shape_broadcast, to_index

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """
        Call the map function.

        Args:
            x: Input tensor
            out: Optional output tensor

        Returns:
            Resulting tensor
        """
        ...


class TensorOps:
    cuda = False

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        raise NotImplementedError()

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        raise NotImplementedError()

    @staticmethod
    def reduce(fn: Callable[[float, float], float], start: float = 0.0) -> Callable[["Tensor", int], "Tensor"]:
        raise NotImplementedError()

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        raise NotImplementedError()

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> MapProto:
        raise NotImplementedError()


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """

        def _map(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            tensor_map(fn)(
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
            )
            return out

        return _map

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """

        def _zip(a: Tensor, b: Tensor) -> Tensor:
            out_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(out_shape)
            tensor_zip(fn)(
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                b._tensor._storage,
                b._tensor._shape,
                b._tensor._strides,
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
            )
            return out

        return _zip

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
        -------
            :class:`TensorData` : new tensor

        """

        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            tensor_reduce(fn)(
                out._tensor._storage,
                out._tensor._shape,
                out._tensor._strides,
                a._tensor._storage,
                a._tensor._shape,
                a._tensor._strides,
                dim,
            )
            return out

        return _reduce

    is_cuda = False


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
    ) -> None:
        in_index = np.zeros(len(in_shape), dtype=np.int32)
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        for i in range(len(out_storage)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(in_storage[in_position])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        b_index = np.zeros(len(b_shape), dtype=np.int32)
        for i in range(len(out_storage)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out_storage: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.zeros(len(out_shape), dtype=np.int32)
        a_index = np.zeros(len(a_shape), dtype=np.int32)
        for i in range(len(out_storage)):
            to_index(i, out_shape, out_index)
            result = out_storage[index_to_position(out_index, out_strides)]
            for j in range(a_shape[reduce_dim]):
                a_index[:] = out_index[:]
                a_index[reduce_dim] = j
                pos = index_to_position(a_index, a_strides)
                result = fn(result, a_storage[pos])
            out_storage[index_to_position(out_index, out_strides)] = result

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
