import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
from dynesty import NestedSampler
from dynesty.utils import merge_runs

from .priors import Uniform, SameAs, Dirac
from .utils import macula

MAX_CORES = cpu_count()

__all__ = ['SpotModel']


class SpotModel(object):
    """Modeler class

    Attributes
    ----------
    t: time array
    y: flux array
    nspots: number of spots
    dy: flux uncertainties (optional)
    Pvec: 2-D array containing rotation period at equator and stellar inclination
    k2: 2nd-order differential rotation coefficient
    k4: 4th-order differential rotation coefficient
    c: stellar limb-darkening coefficients
    d: spot limb-darkening coefficients
    same_limb: whether to always assume c==d in the model
    lon: spot longitudes (rad)
    lat: spot latitudes (rad)
    alpha: spot radius (rad)
    fspot: spot-to-photosphere intensity ratio
    tmax: time of greatest spot area
    life: spot lifetimes
    ingress: spot ingress times
    egress: spot egress times
    U: unspotted surface flux value
    B: instrumental blending factor
    tstart: start time for each of the stitched curves
    tend: end time for each of the stitched curves
    """
    def __init__(self, t, y, nspots, dy=None,
                 Pvec=None, k2=None, k4=None, c=None, d=None, same_limb=False,
                 lon=None, lat=None, alpha=None, fspot=None,
                 tmax=None, life=None, ingress=None, egress=None,
                 U=None, B=None, tstart=None, tend=None):
        self.t = t
        self.y = y
        self.dy = dy
        self.nspots = nspots

        self.Pvec = Pvec
        self.k2 = k2
        self.k4 = k4
        self.c = c
        self.d = d
        self.same_limb = same_limb

        self.lon = lon
        self.lat = lat
        self.alpha = alpha
        self.fspot = fspot
        self.tmax = tmax
        self.life = life
        self.ingress = ingress
        self.egress = egress

        self.U = U
        self.B = B
        self.tstart = tstart
        self.tend = tend

        if self.dy is None:
            self.dy = np.ones_like(self.y)

        # validate star params
        if self.Pvec is None:
            self.Pvec = Uniform(ndim=2, xmin=(0, 0), xmax=(1, 50))
        if self.k2 is None:
            self.k2 = Uniform(-1, 1)
        if self.k4 is None:
            self.k4 = Uniform(-1, 1)

        if self.c is None:
            self.c = Uniform(ndim=4, xmin=(-1, -1, -1, -1), xmax=(1, 1, 1, 1))
        if self.d is None:
            self.d = Uniform(ndim=4, xmin=(-1, -1, -1, -1), xmax=(1, 1, 1, 1))

        if self.same_limb:
            self.d = SameAs(self.c)

        # validate spot params
        baseline = t[-1] - t[0]
        if self.lon is None:
            self.lon = Uniform(ndim=self.nspots, xmin=-np.pi, xmax=np.pi)
        if self.lat is None:
            self.lat = Uniform(ndim=self.nspots, xmin=-np.pi / 2, xmax=np.pi / 2)
        if self.alpha is None:
            self.alpha = Uniform(ndim=self.nspots, xmin=0, xmax=np.pi / 4)
        if self.fspot is None:
            self.fspot = Uniform(ndim=self.nspots, xmin=0, xmax=1)
        if self.tmax is None:
            self.tmax = Uniform(ndim=self.nspots, xmin=t[0], xmax=t[-1])
        if self.life is None:
            self.life = Uniform(ndim=self.nspots, xmin=0, xmax=baseline)
        if self.ingress is None:
            self.ingress = Uniform(ndim=self.nspots, xmin=0, xmax=baseline)
        if self.egress is None:
            self.egress = Uniform(ndim=self.nspots, xmin=0, xmax=baseline)

        # validate inst params
        self.mmax = np.size(self.tstart)
        if self.U is None:
            self.U = Uniform(ndim=self.mmax, xmin=.9, xmax=1.1)
        if self.B is None:
            self.B = Uniform(ndim=self.mmax, xmin=.9, xmax=1.1)

        if self.tstart is None:
            self.tstart = np.array([self.t[0] - .01])
        if self.tend is None:
            self.tend = np.array([self.t[-1] + .01])

        # list of fitted variable names and dictionary of fixed parameter values
        self.fixed_params = {}
        self.fit_names = []
        for key, val in self.parameters.items():
            if isinstance(val, Dirac):
                self.fixed_params[key] = val
            else:
                self.fit_names.append(key)

        # number of prior parameters (unit cube)
        self.ndim = 0
        for v in self.parameters.values():
            self.ndim += v.n_inputs
        # number of loglikelihood parameters (physical)
        self.N = 12 + 8 * self.nspots + 2 * self.mmax

        self.results = {}

    @property
    def parameters(self):
        return {**self.star_pars, **self.spot_pars, **self.inst_pars}

    @property
    def star_pars(self):
        return dict(Pvec=self.Pvec, k2=self.k2, k4=self.k4, c=self.c, d=self.d)

    @property
    def spot_pars(self):
        return dict(lon=self.lon, lat=self.lat, alpha=self.alpha, fspot=self.fspot,
                    tmax=self.tmax, life=self.life, ingress=self.ingress, egress=self.egress)

    @property
    def inst_pars(self):
        return dict(U=self.U, B=self.B)

    def sample(self, x):
        x = np.asarray(x)
        theta = []
        assert x.size == self.ndim, "Dimensionality mismatch"
        i, j = 0, 0
        for key, val in self.parameters.items():
            j += val.n_inputs
            theta = np.append(theta, val.sample(*x[i:j]))
            i = j
        theta[0] = np.arcsin(theta[0])
        return theta

    def predict(self, t, theta):
        """Calculates the model flux for given parameter values

        Parameters
        ----------
        t: array-like with shape (n,)
            time samples where the flux function should be evaluated
        theta: array-like with shape (N,)
            full parameter vector (physical units)

        Returns
        -------
        yf: array-like with shape (n,)
            model flux
        """
        theta = np.asarray(theta)
        assert theta.size == self.N, "Parameter vector with wrong size"
        theta_star = theta[:12]
        theta_spot = theta[12:12 + self.nspots * 8].reshape(8, -1)
        theta_inst = theta[12 + self.nspots * 8:].reshape(2, -1)
        yf = macula(t, theta_star, theta_spot, theta_inst, tstart=self.tstart, tend=self.tend)
        return yf

    def chi(self, theta):
        """Chi squared of parameters given a set of observations

        Parameters
        ----------
        theta: array-like with shape (N,)
            full parameter vector (physical units)

        Returns
        -------
        sse: float
            sum of squared errors weighted by observation uncertainties
        """
        yf = self.predict(self.t, theta)
        sse = np.sum(np.square((yf - self.y) / self.dy))
        return sse

    def loglike(self, x):
        n = self.t.size
        c = - (n * np.log(2 * np.pi) - np.log(self.dy).sum()) / 2
        return c - self.chi(x) / 2

    def reduced_chi(self, theta):
        nu = self.t.size - self.ndim
        return self.chi(theta) / nu

    def multinest(self, sampling_efficiency=0.01, const_efficiency_mode=True,
                  n_live_points=4000, **kwargs):
        def prior(cube):
            return cube

        def logl(cube):
            theta = self.sample(cube)
            n = self.t.size
            c = - .5 * n * np.log(2 * np.pi) - .5 * np.log(self.dy).sum()
            return c - .5 * self.chi(theta)

        results = solve(LogLikelihood=logl, Prior=prior, n_dims=self.ndim, sampling_efficiency=sampling_efficiency,
                        const_efficiency_mode=const_efficiency_mode, n_live_points=n_live_points, **kwargs)
        return results

    def run(self, nlive=1000, cores=None, filename=None, **kwargs):
        merge = 'no'
        if filename is not None and os.path.isfile(filename):
            doit = input(f'There seems to be a file named {filename}. '
                         f'Would you like to run anyway? [y/n] ').lower()
            if doit in ['no', 'n']:
                with open(filename, 'br') as file:
                    self.results = pickle.load(file)
                return
        if cores is None or cores > MAX_CORES:
            cores = MAX_CORES
        try:
            with Pool(cores) as pool:
                sampler = NestedSampler(self.loglike, self.sample, self.N, npdim=self.ndim,
                                        nlive=nlive, pool=pool, queue_size=cores, **kwargs)
                sampler.run_nested()
        except KeyboardInterrupt:
            pass
        if filename is not None and os.path.isfile(filename):
            merge = input('Merge new run with previous data? [y/n] ').lower()
        if merge in ['no', 'n']:
            self.results = sampler.results
        else:
            with open(filename, 'br') as file:
                res = pickle.load(file)
            self.results = merge_runs([sampler.results, res])
        if filename is not None:
            with open(filename, 'bw') as file:
                pickle.dump(self.results, file)
