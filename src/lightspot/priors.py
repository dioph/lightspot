import numpy as np
from scipy.stats import norm, truncnorm

from .utils import polygon_intersection, triangle_area, triangulate


class Prior(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def __call__(self, x):
        return x


class Duplicate(Prior):
    def __init__(self, prior):
        super(Duplicate, self).__init__(prior.n_inputs, 2 * prior.n_outputs)
        self.prior = prior

    def __call__(self, x):
        out = self.prior(x)
        return np.hstack([out, out])


class Stack(Prior):
    def __init__(self, prior1, prior2):
        if prior1.n_inputs != prior2.n_inputs:
            raise ValueError("Can't stack priors with different number of inputs!")
        super(Stack, self).__init__(
            prior1.n_inputs, prior1.n_outputs + prior2.n_outputs
        )
        self.prior1 = prior1
        self.prior2 = prior2

    def __call__(self, x):
        out1 = self.prior1(x)
        out2 = self.prior2(x)
        return np.hstack[out1, out2]


# UNIVARIATE DISTRIBUTIONS


class Dirac(Prior):
    def __init__(self, x0):
        x0 = np.atleast_1d(x0)
        if x0.ndim > 1:
            raise ValueError("x0 must be at most 1-D")
        super(Dirac, self).__init__(0, x0.shape[0])
        self.x0 = x0

    def __call__(self, dummy):
        return self.x0


class Uniform(Prior):
    def __init__(self, xmin=0.0, xmax=1.0, ndim=1):
        super(Uniform, self).__init__(ndim, ndim)
        self.xmin = np.asarray(xmin)
        self.xmax = np.asarray(xmax)

    def __call__(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"Got {x.shape[1]} inputs, expected {self.n_inputs}.")
        self.last_sampled = x * (self.xmax - self.xmin) + self.xmin
        return self.last_sampled


class SineUniform(Prior):
    def __init__(self, sinxmin=0.0, sinxmax=1.0, ndim=1):
        super(SineUniform, self).__init__(ndim, ndim)
        self.sinxmin = np.asarray(sinxmin)
        self.sinxmax = np.asarray(sinxmax)

    def __call__(self, sinx):
        sinx = np.atleast_2d(sinx)
        if sinx.shape[1] != self.n_inputs:
            raise ValueError(f"Got {sinx.shape[1]} inputs, expected {self.n_inputs}.")
        return np.arcsin(sinx * (self.sinxmax - self.sinxmin) + self.sinxmin)


class LogUniform(Prior):
    def __init__(self, logxmin=0.0, logxmax=1.0, ndim=1):
        super(LogUniform, self).__init__(ndim, ndim)
        self.logxmin = np.asarray(logxmin)
        self.logxmax = np.asarray(logxmax)

    def __call__(self, logx):
        logx = np.atleast_2d(logx)
        if logx.shape[1] != self.n_inputs:
            raise ValueError(f"Got {logx.shape[1]} inputs, expected {self.n_inputs}.")
        return np.exp(logx * (self.logxmax - self.logxmin) + self.logxmin)


class Normal(Prior):
    def __init__(self, mu=0.0, sd=1.0, ndim=1):
        super(Normal, self).__init__(ndim, ndim)
        self.mu = np.asarray(mu)
        self.sd = np.asarray(sd)

    def __call__(self, q):
        q = np.atleast_2d(q)
        if q.shape[1] != self.n_inputs:
            raise ValueError(f"Got {q.shape[1]} inputs, expected {self.n_inputs}.")
        return norm.ppf(q, self.mu, self.sd)


class TruncNormal(Prior):
    def __init__(self, mu=0.0, sd=1.0, xmin=0.0, xmax=1.0, ndim=1):
        super(TruncNormal, self).__init__(ndim, ndim)
        self.mu = np.asarray(mu)
        self.sd = np.asarray(sd)
        self.xmin = np.asarray(xmin)
        self.xmax = np.asarray(xmax)
        self.xmin = ((self.xmin - self.mu) / self.sd,)
        self.xmax = (self.xmax - self.mu) / self.sd

    def __call__(self, q):
        q = np.atleast_2d(q)
        if q.shape[1] != self.n_inputs:
            raise ValueError(f"Got {q.shape[1]} inputs, expected {self.n_inputs}.")
        return truncnorm.ppf(q, self.xmin, self.xmax, self.mu, self.sd)


class LogNormal(Prior):
    def __init__(self, logmu=0.0, logsd=1.0, ndim=1):
        super(LogNormal, self).__init__(ndim, ndim)
        self.logmu = np.asarray(logmu)
        self.logsd = np.asarray(logsd)

    def __call__(self, q):
        q = np.atleast_2d(q)
        if q.shape[1] != self.n_inputs:
            raise ValueError(f"Got {q.shape[1]} inputs, expected {self.n_inputs}.")
        return np.exp(norm.ppf(q, self.logmu, self.logsd))


# BIVARIATE DISTRIBUTIONS


class Triangular(Prior):
    def __init__(self, triangle):
        super(Triangular, self).__init__(2, 2)
        self.triangle = triangle.reshape(-1, 3, 2)

    def __call__(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            raise ValueError(f"Got {u.shape[1]} inputs, expected {self.n_inputs}.")
        q1, q2 = u.T
        q1 = q1.reshape(-1, 1)
        q2 = q2.reshape(-1, 1)
        A = self.triangle[:, 0, :]
        B = self.triangle[:, 1, :]
        C = self.triangle[:, 2, :]
        return (1 - np.sqrt(q1)) * A + np.sqrt(q1) * (1 - q2) * B + q2 * np.sqrt(q1) * C


class Polygon(Prior):
    def __init__(self, poly):
        super(Polygon, self).__init__(2, 2)
        self.triangles = triangulate(poly)
        area = np.sum([triangle_area(t) for t in self.triangles])
        self.relative_area = np.array([triangle_area(t) / area for t in self.triangles])
        area_limits = np.cumsum(self.relative_area)
        self.area_limits = np.append(0, area_limits)

    def __call__(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            raise ValueError(f"Got {u.shape[1]} inputs, expected {self.n_inputs}.")
        q1, q2 = u.T
        q1 = q1.reshape(-1, 1)
        q2 = q2.reshape(-1, 1)
        group = np.searchsorted(self.area_limits, q2) - 1
        triangle = self.triangles[group]
        q2 = (q2 - self.area_limits[group]) / self.relative_area[group]
        return Triangular(triangle)(np.hstack([q1, q2]))


class ConstrainPvec(Prior):
    def __init__(
        self,
        pmin=0,
        pmax=50,
        sinimin=0,
        sinimax=1,
        vmin=1e-9,
        vmax=1e9,
        rmin=1e-9,
        rmax=1e9,
    ):
        super(ConstrainPvec, self).__init__(2, 2)
        k = 2 * np.pi / 86400
        wmin, wmax = np.array([vmin / rmax, vmax / rmin]) / 695700
        poly1 = np.array([[0, 0], [1, k / wmin], [1, k / wmax]])
        poly2 = np.array(
            [[sinimin, pmin], [sinimin, pmax], [sinimax, pmax], [sinimax, pmin]]
        )
        self.poly = Polygon(polygon_intersection(poly1, poly2))

    def __call__(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            raise ValueError(f"Got {u.shape[1]} inputs, expected {self.n_inputs}.")
        sini, peq = self.poly(u).T
        i = np.arcsin(sini)
        return np.vstack(i, peq).T


# LDC DISTRIBUTIONS


class QuadraticLD(Prior):
    def __init__(self, amin=0.0, amax=2.0, bmin=-1.0, bmax=1.0):
        super(QuadraticLD, self).__init__(2, 4)
        poly1 = np.array([[amin, bmin], [amin, bmax], [amax, bmax], [amax, bmin]])
        poly2 = np.array([[0, 0], [0, 1], [2, -1]])
        self.poly = polygon_intersection(poly1, poly2)

    def __call__(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            raise ValueError(f"Got {u.shape[1]} inputs, expected {self.n_inputs}.")
        a, b = Polygon(self.poly)(u).T
        c1 = np.zeros_like(a)
        c2 = a + 2 * b
        c3 = np.zeros_like(a)
        c4 = -b
        return np.vstack([c1, c2, c3, c4]).T


class ThreeParamLD(Prior):
    def __init__(self):
        super(ThreeParamLD, self).__init__(3, 4)

    def __call__(self, u):
        u = np.atleast_2d(u)
        if u.shape[1] != self.n_inputs:
            raise ValueError(f"Got {u.shape[1]} inputs, expected {self.n_inputs}.")
        q1, q2, q3 = u.T
        q1 = q1.reshape(-1, 1)
        q2 = q2.reshape(-1, 1)
        q3 = q3.reshape(-1, 1)
        c1 = np.zeros_like(q1)
        c2 = (q1 ** (1 / 3) / 12) * (
            28 * (9 - 5 * np.sqrt(2))
            + 3
            * np.sqrt(q2)
            * (
                -6 * np.cos(2 * np.pi * q3)
                + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))
            )
        )
        c3 = (q1 ** (1 / 3) / 9) * (
            -632
            + 396 * np.sqrt(2)
            + 3 * np.sqrt(q2) * (4 - 21 * np.sqrt(2)) * np.sin(2 * np.pi * q3)
        )
        c4 = (q1 ** (1 / 3) / 12) * (
            28 * (9 - 5 * np.sqrt(2))
            + 3
            * np.sqrt(q2)
            * (
                6 * np.cos(2 * np.pi * q3)
                + (3 + 10 * np.sqrt(2) * np.sin(2 * np.pi * q3))
            )
        )
        return np.hstack([c1, c2, c3, c4])
