from typing import Callable, Iterable, Tuple
from . import operators


class MathTest:
    @staticmethod
    def neg(a: float) -> float:
        """Negate the argument"""
        return operators.neg(a)

    @staticmethod
    def addConstant(a: float) -> float:
        """Add constant to the argument"""
        return operators.add(a, 5.0)

    @staticmethod
    def square(a: float) -> float:
        """Manual square"""
        return operators.mul(a, a)

    @staticmethod
    def cube(a: float) -> float:
        """Manual cube"""
        return operators.mul(operators.mul(a, a), a)

    @staticmethod
    def subConstant(a: float) -> float:
        """Subtract a constant from the argument"""
        return operators.add(a, -5.0)

    @staticmethod
    def multConstant(a: float) -> float:
        """Multiply a constant to the argument"""
        return operators.mul(a, 5.0)

    @staticmethod
    def div(a: float) -> float:
        """Divide by a constant"""
        return operators.mul(a, 0.2)

    @staticmethod
    def inv(a: float) -> float:
        """Invert after adding"""
        return operators.inv(operators.add(a, 1.0))

    @staticmethod
    def sig(a: float) -> float:
        """Apply sigmoid"""
        return operators.sigmoid(a)

    @staticmethod
    def log(a: float) -> float:
        """Apply log to a large value"""
        return operators.log(operators.add(a, 100.0))

    @staticmethod
    def relu(a: float) -> float:
        """Apply relu"""
        return operators.relu(a)

    @staticmethod
    def exp(a: float) -> float:
        """Apply exp to a smaller value"""
        return operators.exp(operators.mul(a, 0.1))

    @staticmethod
    def add2(a: float, b: float) -> float:
        """Add two arguments"""
        return operators.add(a, b)

    @staticmethod
    def mul2(a: float, b: float) -> float:
        """Mul two arguments"""
        return operators.mul(a, b)

    @staticmethod
    def div2(a: float, b: float) -> float:
        """Divide two arguments"""
        return operators.mul(a, operators.inv(b))

    @classmethod
    def _tests(
        cls,
    ) -> Tuple[
        list[Tuple[str, Callable[[float], float]]],
        list[Tuple[str, Callable[[float, float], float]]],
        list[Tuple[str, Callable[[Iterable[float]], float]]],
    ]:
        """Returns a list of all the math tests."""
        one_arg = [
            ("neg", cls.neg),
            ("addConstant", cls.addConstant),
            ("square", cls.square),
            ("cube", cls.cube),
            ("subConstant", cls.subConstant),
            ("multConstant", cls.multConstant),
            ("div", cls.div),
            ("inv", cls.inv),
            ("sig", cls.sig),
            ("log", cls.log),
            ("relu", cls.relu),
            ("exp", cls.exp),
        ]
        two_arg = [("add2", cls.add2), ("mul2", cls.mul2), ("div2", cls.div2)]
        return (one_arg, two_arg, [])


class MathTestVariable(MathTest):
    pass
