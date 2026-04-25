import cupy as cp

from .gmacula import macula
from .sampler import SpotModel

__all__ = ["GPUSpotModel"]


class GPUSpotModel(SpotModel):
    def __init__(self, t, y, nspots, dy=None, priors=None, tstart=None, tend=None):
        super(GPUSpotModel, self).__init__(t, y, nspots, dy, priors, tstart, tend)
        self.func = macula
        self.t_gpu = cp.asarray(self.t)
        self.y_gpu = cp.asarray(self.y)
        self.dy_gpu = cp.asarray(self.dy)
        self.nlog2pi = self.t_gpu.size * cp.log(2 * cp.pi)
        self.tstart_gpu = cp.asarray(self.tstart, dtype=self._dtype)
        self.tend_gpu = cp.asarray(self.tend, dtype=self._dtype)

    def predict(self, t, theta):
        """Calculates the model flux for given parameter values

        Parameters
        ----------
        t: array-like with shape (ndata,)
            time samples where the flux function should be evaluated
        theta: array-like with shape (jmax,)
            full parameter vector (physical units)

        Returns
        -------
        yf: array-like with shape (ndata,)
            model flux
        """
        theta_params = cp.atleast_2d(theta)[..., :-1].astype(self._dtype)
        if theta_params.shape[1] != (self.jmax - 1):
            raise ValueError("Parameter vector with wrong size.")
        yf = self.func(t, theta_params, self.tstart_gpu, self.tend_gpu)
        return yf

    def eff_var(self, theta):
        theta_gpu = cp.atleast_2d(theta).astype(self._dtype)
        jitter = theta_gpu[..., -1]
        eff_var = self.dy_gpu**2 + jitter[:, None] ** 2
        return eff_var

    def chi(self, theta):
        """Chi squared of parameters given a set of observations

        Parameters
        ----------
        theta: array-like with shape (jmax,)
            Full parameter vector (physical units).

        Returns
        -------
        sse: float
            Sum of squared errors weighted by observation uncertainties.
        """
        theta_gpu = cp.atleast_2d(theta).astype(self._dtype)
        jitter = theta_gpu[..., -1]
        theta_params = theta_gpu[..., :-1]
        eff_var = self.dy_gpu**2 + jitter[:, None] ** 2
        if theta_params.shape[1] != (self.jmax - 1):
            raise ValueError("Parameter vector with wrong size.")
        yf = self.func(self.t_gpu, theta_params, self.tstart_gpu, self.tend_gpu)
        sse = cp.sum((yf - self.y_gpu) ** 2 / eff_var, axis=1)
        return sse.get()

    def loglike(self, theta):
        theta_gpu = cp.atleast_2d(theta).astype(self._dtype)
        theta_params = cp.atleast_2d(theta)[..., :-1].astype(self._dtype)
        jitter = theta_gpu[..., -1]
        eff_var = self.dy_gpu**2 + jitter[:, None] ** 2
        norm_c = -(self.nlog2pi - cp.log(eff_var).sum(axis=1)) / 2
        if theta_params.shape[1] != (self.jmax - 1):
            raise ValueError("Parameter vector with wrong size.")
        yf = self.func(self.t_gpu, theta_params, self.tstart_gpu, self.tend_gpu)
        sse = cp.sum((yf - self.y_gpu) ** 2 / eff_var, axis=1)
        return (norm_c - sse / 2).get()
