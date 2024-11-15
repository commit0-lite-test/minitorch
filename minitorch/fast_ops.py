from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional
import numpy as np
from numba import njit, prange
from .tensor_data import broadcast_index, index_to_position, shape_broadcast, to_index
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""

        def _map(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            tensor_map(fn)(
                a.storage(),
                a.shape,
                a.strides(),
                out.storage(),
                out.shape,
                out.strides(),
            )
            return out

        return _map

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""

        def _zip(a: Tensor, b: Tensor) -> Tensor:
            out_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(out_shape)
            tensor_zip(fn)(
                a.storage(),
                a.shape,
                a.strides(),
                b.storage(),
                b.shape,
                b.strides(),
                out.storage(),
                out.shape,
                out.strides(),
            )
            return out

        return _zip

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""

        def _reduce(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            tensor_reduce(fn)(
                out.storage(),
                out.shape,
                out.strides(),
                a.storage(),
                a.shape,
                a.strides(),
                dim,
            )
            return out

        return _reduce

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Shape validation
        assert a.shape[-1] == b.shape[-2], "Invalid shapes for matrix multiplication"

        # Determine output shape
        out_shape = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        out_shape.extend([a.shape[-2], b.shape[-1]])

        # Create output tensor
        out = a.zeros(tuple(out_shape))

        # Perform matrix multiplication
        tensor_matrix_multiply(
            out.storage(),
            out.shape,
            out.strides(),
            a.storage(),
            a.shape,
            a.strides(),
            b.storage(),
            b.shape,
            b.strides(),
        )

        return out


@njit(parallel=True)
def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

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
        for i in prange(len(out_storage)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            in_position = index_to_position(in_index, in_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(in_storage[in_position])

    return _map


@njit(parallel=True)
def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

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
        for i in prange(len(out_storage)):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)
            out_position = index_to_position(out_index, out_strides)
            out_storage[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


@njit(parallel=True)
def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

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
        for i in prange(len(out_storage)):
            to_index(i, out_shape, out_index)
            result = out_storage[index_to_position(out_index, out_strides)]
            for j in range(a_shape[reduce_dim]):
                a_index[:] = out_index[:]
                a_index[reduce_dim] = j
                pos = index_to_position(a_index, a_strides)
                result = fn(result, a_storage[pos])
            out_storage[index_to_position(out_index, out_strides)] = result

    return _reduce


@njit(parallel=True, fastmath=True)
def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if len(a_shape) == 3 else 0
    b_batch_stride = b_strides[0] if len(b_shape) == 3 else 0

    for i in prange(out_shape[0]):  # Batch dimension
        for j in prange(out_shape[1]):  # Rows of output
            for k in range(out_shape[2]):  # Columns of output
                a_index = i * a_batch_stride + j * a_strides[-2]
                b_index = i * b_batch_stride + k * b_strides[-1]
                out_index = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]

                acc = 0.0
                for m in range(a_shape[-1]):
                    acc += (
                        a_storage[a_index + m * a_strides[-1]]
                        * b_storage[b_index + m * b_strides[-2]]
                    )
                out[out_index] = acc


tensor_matrix_multiply = njit(parallel=True, fastmath=True)(_tensor_matrix_multiply)
