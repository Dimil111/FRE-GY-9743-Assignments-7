import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Continuous interpolator through the IFR points that is monotone and convex on every node interval.

    On [x_i, x_{i+1}] with secant s, the derivative runs linearly from a_i to s and then from s to b_{i+1},
    where a_i <= s <= b_{i+1}. The curve is therefore a convex quadratic spline with one extra knot per
    interval, and its integral is closed form. Node slopes are harmonic means of the adjacent secants
    (zero where the secants change sign). Where the secants decrease, the node has separate left and right
    slopes equal to the neighbouring secants, so the curve is continuous there but not differentiable.
    Extrapolation is flat.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        self.x_ = np.asarray(axis1, dtype=float)
        self.y_ = np.asarray(values, dtype=float)
        assert np.all(np.diff(self.x_) > 0)
        self._build_slopes()
        self._build_integrals()

    def _build_slopes(self) -> None:
        n = self.length_
        self.h_ = np.diff(self.x_)
        self.s_ = np.diff(self.y_) / self.h_ if n > 1 else np.zeros(0)
        s = self.s_
        a, b = np.zeros(max(n - 1, 0)), np.zeros(max(n - 1, 0))
        for i in range(1, n - 1):
            left, right = s[i - 1], s[i]
            if left <= right:
                m = 0.0 if left * right <= 0.0 else 2.0 * left * right / (left + right)
                b[i - 1], a[i] = m, m
            else:
                b[i - 1], a[i] = left, right
        if n == 2:
            a[0], b[0] = s[0], s[0]
        elif n > 2:
            a[0] = 2.0 * s[0] - b[0]
            b[-1] = 2.0 * s[-1] - a[-1]
            if s[0] > 0.0:
                a[0] = max(a[0], 0.0)
            if s[-1] < 0.0:
                b[-1] = min(b[-1], 0.0)
        self.a_, self.b_ = a, b
        with np.errstate(divide="ignore", invalid="ignore"):
            self.xi_ = np.where(b > a, (b - s) / (b - a), 0.0)

    def _piece(self, i: int, u: float):
        """Value and integral from x_i to x_i + u inside interval i."""
        a, b, s, h, xi, y0 = self.a_[i], self.b_[i], self.s_[i], self.h_[i], self.xi_[i], self.y_[i]
        h1 = xi * h
        if u <= h1:
            slope_rate = (s - a) / h1 if h1 > 0.0 else 0.0
            return y0 + a * u + 0.5 * slope_rate * u * u, y0 * u + 0.5 * a * u * u + slope_rate * u ** 3 / 6.0
        y1 = y0 + a * h1 + 0.5 * (s - a) * h1
        i1 = y0 * h1 + 0.5 * a * h1 * h1 + (s - a) * h1 * h1 / 6.0
        v = u - h1
        rate = (b - s) / (h - h1) if h - h1 > 0.0 else 0.0
        return y1 + s * v + 0.5 * rate * v * v, i1 + y1 * v + 0.5 * s * v * v + rate * v ** 3 / 6.0

    def _build_integrals(self) -> None:
        cum = np.zeros(self.length_)
        for i in range(self.length_ - 1):
            cum[i + 1] = cum[i] + self._piece(i, self.h_[i])[1]
        self.cum_ = cum

    def _interval(self, x: float) -> int:
        return int(min(max(np.searchsorted(self.x_, x, side='left') - 1, 0), self.length_ - 2))

    def interpolate(self, x: float) -> float:
        if x <= self.x_[0]:
            return float(self.y_[0])
        if x >= self.x_[-1]:
            return float(self.y_[-1])
        i = self._interval(x)
        return float(self._piece(i, x - self.x_[i])[0])

    def _cumulative(self, x: float) -> float:
        if x <= self.x_[0]:
            return float(self.y_[0] * (x - self.x_[0]))
        if x >= self.x_[-1]:
            return float(self.cum_[-1] + self.y_[-1] * (x - self.x_[-1]))
        i = self._interval(x)
        return float(self.cum_[i] + self._piece(i, x - self.x_[i])[1])

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._cumulative(end_x) - self._cumulative(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
