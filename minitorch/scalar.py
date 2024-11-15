from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, Union
from .autodiff import Context, backpropagate, central_difference
from .scalar_functions import EQ, LT, Add, Inv, Mul, Neg, ScalarFunction

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


_var_count = 0


class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.
    """

    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(
        self,
        v: float,
        back: ScalarHistory = ScalarHistory(),
        name: Optional[str] = None,
    ):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)  # type: ignore

    def __bool__(self) -> bool:
        return bool(self.data)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(self, b)  # type: ignore

    def __gt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(b, self)  # type: ignore

    def __eq__(self, b: ScalarLike) -> Scalar:
        return EQ.apply(self, b)  # type: ignore

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, Neg.apply(b))  # type: ignore

    def __neg__(self) -> Scalar:
        return Neg.apply(self)  # type: ignore

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b  # type: ignore

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b  # type: ignore

    def accumulate_derivative(self, x: Any) -> None:
        """Add `x` to the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x: value to be accumulated

        """
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is None or self.history.last_fn is None

    def backward(self, d_output: Optional[float] = None) -> None:
        """Calls autodiff to fill in the derivatives for the history of this object.

        Args:
        ----
            d_output (number, opt): starting derivative to backpropagate through the model
                                   (typically left out, and assumed to be 1.0).

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a python function.
    Asserts False if derivative is incorrect.

    Args:
    ----
        f: function from n-scalars to 1-scalar.
        *scalars: n input scalar values.

    """
    out = f(*scalars)
    out.backward()

    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        assert abs(x.derivative - check) < 1e-2, (
            f"Derivative of {f.__name__} with respect to argument {i} is incorrect. "
            f"Expected {check}, got {x.derivative}"
        )
