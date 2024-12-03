from __future__ import annotations
from typing import TYPE_CHECKING, Any, Tuple, Union
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from .scalar import Scalar, ScalarLike

if TYPE_CHECKING:
    from .scalar import ScalarHistory


def wrap_tuple(x: Any) -> Tuple[Any, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x: Tuple[Any, ...]) -> Any:
    """Turn a singleton tuple into a value"""
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @staticmethod
    def forward(ctx: Context, *inputs: float) -> float:
        """Forward function, to be implemented by subclasses"""
        raise NotImplementedError()

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Backward function, to be implemented by subclasses"""
        raise NotImplementedError()

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Union[Scalar, Tuple[Scalar, ...]]:
        """Apply the function to values"""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(Scalar(v))
                raw_vals.append(v)

        ctx = Context()
        out = cls.forward(ctx, *raw_vals)
        if not isinstance(out, tuple):
            out = (out,)

        result = tuple(Scalar(o) for o in out)
        if len(result) == 1:
            result = result[0]

        if len(scalars) > 0 and isinstance(result, Scalar):
            result.history = ScalarHistory(cls, ctx, scalars)

        return result


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for addition."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for addition."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for logarithm."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for logarithm."""
        (a,) = ctx.saved_values
        return (operators.log_back(a, d_output),)


class Mul(ScalarFunction):
    """Multiplication function"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication."""
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication."""
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse."""
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for inverse."""
        (a,) = ctx.saved_values
        return (operators.inv_back(a, d_output),)


class Neg(ScalarFunction):
    """Negation function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation."""
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for negation."""
        return (-d_output,)


class Sigmoid(ScalarFunction):
    """Sigmoid function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid."""
        sigmoid_val = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for sigmoid."""
        (sigmoid_val,) = ctx.saved_values
        return (d_output * sigmoid_val * (1 - sigmoid_val),)


class ReLU(ScalarFunction):
    """ReLU function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for ReLU."""
        (a,) = ctx.saved_values
        return (operators.relu_back(a, d_output),)


class Exp(ScalarFunction):
    """Exp function"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential."""
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        """Backward pass for exponential."""
        (exp_val,) = ctx.saved_values
        return (d_output * exp_val,)


class LT(ScalarFunction):
    """Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less-than."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less-than."""
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality."""
        return 0.0, 0.0
class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return cls.backward(ctx, d_out)  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> "Scalar":
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = ctx.saved_values

        return Scalar(c, History(cls, ctx, scalars, back))

class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Mul(ScalarFunction):
    "Multiplication function $f(x, y) = x * y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        # TODO: Implement for Task 1.4.
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Neg(ScalarFunction):
    "Negation function $f(x) = -x$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        # TODO: Implement for Task 1.4.
        return (operators.neg(d_output),)


class Inv(ScalarFunction):
    "Inverse function $f(x) = 1/x$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # TODO: Implement for Task 1.2.
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        # TODO: Implement for Task 1.4.
        (a,) = ctx.saved_values
        return (operators.neg(operators.inv(operators.mul(a, a)) * d_output),)


class LT(ScalarFunction):
    "Less-than function $f(x) =$ 1.0 if x is less than y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0


class EQ(ScalarFunction):
    "Equal function $f(x) =$ 1.0 if x is equal to y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        # TODO: Implement for Task 1.2.
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        # TODO: Implement for Task 1.4.
        return 0.0, 0.0

def derivative_check(f: Callable[[Scalar], Scalar], *scalars: Scalar) -> None:
    # TODO: Implement for Task 1.5.
    raise NotImplementedError("Implement for Task 1.5")
