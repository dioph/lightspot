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
        yf = self.predict(self.t_gpu, theta)
        sse = cp.sum(cp.square((yf - self.y_gpu) / self.dy_gpu), axis=1)
        return sse.get()
