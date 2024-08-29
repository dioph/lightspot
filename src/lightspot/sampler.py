from multiprocessing import Pool, cpu_count
import os
import pickle

from dynesty import NestedSampler
from dynesty.utils import merge_runs
import numpy as np
from ultranest import ReactiveNestedSampler, stepsampler
from ultranest.plot import PredictionBand

from .macula import macula
from .priors import QuadraticLD, SineUniform, Uniform

MAX_CORES = cpu_count()

__all__ = ["NestedSolver"]


class NestedSolver(object):
    def __init__(self, defaults, t, y, dy=None, priors=None):
        self.t = t
        self.y = y.astype(t.dtype)
        if dy is None:
            dy = np.ones_like(self.y)
        self.dy = dy.astype(self.y.dtype)
        self.norm_c = -(self.t.size * np.log(2 * np.pi) - np.log(self.dy).sum()) / 2
        self.defaults = defaults
        self.param_names = list(self.defaults.keys())
        param_id_bounds = np.append(
            0, np.cumsum([prior.n_outputs for prior in defaults.values()])
        )
        self.jmax = param_id_bounds[-1]
        param_ids = {}
        for i in range(len(defaults)):
            param_ids[self.param_names[i]] = np.r_[
                param_id_bounds[i] : param_id_bounds[i + 1]
            ]
        self.param_ids = param_ids
        self.priors = self._validate_params(priors)
        self.id_priors = {}
        for key, val in self.priors.items():
            if isinstance(key, str):
                id_mask = param_ids[key]
            else:
                id_mask = np.hstack([param_ids[k] for k in key])
            self.id_priors[tuple(id_mask)] = val
        self.ndim = sum([prior.n_inputs for prior in self.priors.values()])
        # define self.wrap
        i, j = 0, 0
        self.wrap = np.empty(self.ndim)
        for key, val in self.id_priors.items():
            j += val.n_inputs
            self.wrap[i:j] = val.wrap
            i = j
        # list of fitted variable names and dictionary of fixed parameter values
        self.fit_names = []
        self.fixed_params = {}
        # number of prior parameters (unit cube)
        i = 0
        for key, val in self.priors.items():
            if val.n_inputs == 0:
                self.fixed_params[key] = val
            else:
                # TODO
                for j in range(val.n_inputs):
                    # self.fit_names.append(val.input_names[j])
                    self.fit_names.append(f"{str(key)[:20]}_{{{i}}}")
                    i += 1

    def _validate_params(self, priors):
        if priors is None:
            return self.defaults
        # check if all given keys are valid
        for key in priors.keys():
            if not all(np.in1d(key, self.param_names)):
                raise KeyError(key)
        # TODO: check if all given keys are unique
        # check if the number of outputs is conserved
        for key in priors.keys():
            if isinstance(key, str):
                n_outputs = self.defaults[key].n_outputs
            else:
                n_outputs = sum(
                    [prior.n_outputs for k, prior in self.defaults.items() if k in key]
                )
            if n_outputs != priors[key].n_outputs:
                raise ValueError(
                    f"The prior distribution for the combination {key} should"
                    f" return {n_outputs} values ({priors[key].n_outputs} given)."
                )
        # replace default keys by the ones given
        full_priors = self.defaults.copy()
        for name in self.param_names:
            for key in priors.keys():
                if isinstance(key, str):
                    if name == key:
                        full_priors.pop(name)
                else:
                    if name in key:
                        full_priors.pop(name)
        full_priors.update(priors)
        return full_priors

    def _latexify(self, name, i=None):
        parts = name.split("_")
        if i is not None:
            parts.append(str(i))
        for j in range(len(parts)):
            if parts[j].lower() in [
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
                "iota",
                "kappa",
                "lambda",
                "mu",
                "nu",
                "xi",
                "omicron",
                "pi",
                "rho",
                "sigma",
                "tau",
                "upsilon",
                "phi",
                "chi",
                "psi",
                "omega",
            ]:
                parts[j] = "\\" + parts[j]
        name = "${{" + "}}_{{".join(parts) + "}}$"
        return name

    def predict(self, t, theta):
        """Base method to be overridden by the actual model.
        Should be vectorized, i.e., return a 2d ndarray with shape
        (theta.shape[0], t.size)"""
        return np.atleast2d(self.y)

    def prior_transform(self, cube):
        cube = np.atleast_2d(cube)
        theta = np.empty((cube.shape[0], self.jmax), dtype=self.t.dtype)
        if cube.shape[1] != self.ndim:
            raise ValueError("Dimensionality mismatch")
        i, j = 0, 0
        for key, val in self.id_priors.items():
            j += val.n_inputs
            theta[:, list(key)] = val(cube[:, i:j])
            i = j
        return theta

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
        yf = self.predict(self.t, theta)
        sse = np.sum(np.square((yf - self.y) / self.dy), axis=1)
        return sse

    def loglike(self, theta):
        return self.norm_c - self.chi(theta) / 2

    def reduced_chi(self, theta):
        nu = self.t.size - self.ndim
        return self.chi(theta) / nu

    def minimize(self):
        # TODO
        pass

    def mcmc(self, n_walkers, n_steps, burn, use_prior=True):
        # TODO
        pass

    def nested_sample(self, log_dir=None, n_slice=0, **kwargs):
        def logl(cube):
            theta = self.prior_transform(cube)
            return self.loglike(theta)

        self.sampler = ReactiveNestedSampler(
            self.fit_names,
            logl,
            log_dir=log_dir,
            resume=resume,
            wrapped_params=self.wrap,
            vectorized=True,
        )
        if n_slice > 0:
            self.sampler.stepsampler = stepsampler.RegionSliceSampler(nsteps=n_slice)
        results = self.sampler.run(**kwargs)
        results = self._post_processing(results)
        self.sampler.run_sequence["samples"] = results["weighted_samples"]["points"]
        return results

    def run(self, nlive=1000, cores=None, filename=None, **kwargs):
        merge = "no"
        if filename is not None and os.path.isfile(filename):
            doit = input(
                f"There seems to be a file named {filename}. "
                f"Would you like to run anyway? [y/n] "
            ).lower()
            if doit in ["no", "n"]:
                with open(filename, "br") as file:
                    self.dynesty_results = pickle.load(file)
                return
        if cores is None or cores > MAX_CORES:
            cores = MAX_CORES
        try:
            with Pool(cores) as pool:
                sampler = NestedSampler(
                    self.loglike,
                    self.prior_transform,
                    self.jmax,
                    npdim=self.ndim,
                    nlive=nlive,
                    pool=pool,
                    queue_size=cores,
                    **kwargs,
                )
                sampler.run_nested()
        except KeyboardInterrupt:
            pass
        if filename is not None and os.path.isfile(filename):
            merge = input("Merge new run with previous data? [y/n] ").lower()
        if merge in ["no", "n"]:
            self.dynesty_results = sampler.results
        else:
            with open(filename, "br") as file:
                res = pickle.load(file)
            self.dynesty_results = merge_runs([sampler.results, res])
        if filename is not None:
            with open(filename, "bw") as file:
                pickle.dump(self.dynesty_results, file)

    def _post_processing(self, results):
        upoints = results["weighted_samples"]["upoints"]
        points = self.prior_transform(upoints)
        results["weighted_samples"]["points"] = points
        samples = self.prior_transform(results["samples"])
        results["samples"] = samples
        results["posterior"]["mean"] = samples.mean(axis=0).tolist()
        results["posterior"]["stdev"] = samples.std(axis=0).tolist()
        results["posterior"]["median"] = np.percentile(samples, 50, axis=0).tolist()
        results["posterior"]["errlo"] = np.percentile(samples, 15.8655, axis=0).tolist()
        results["posterior"]["errup"] = np.percentile(samples, 84.1345, axis=0).tolist()
        results["maximum_likelihood"]["point"] = self.prior_transform(
            results["maximum_likelihood"]["point"]
        ).tolist()
        return results

    def plot_fit(self, t_grid, color="k"):
        band = PredictionBand(t_grid)
        for theta in self.sampler.results["samples"]:
            band.add(self.predict(t_grid, theta).flatten())
        band.line(color=color)
        band.shade(color=color, alpha=0.3)
        band.shade(q=0.49, color=color, alpha=0.2)


class SpotModel(NestedSolver):
    def __init__(self, t, y, nspots, dy=None, priors=None, tstart=None, tend=None):
        self.nspots = nspots
        if tstart is None:
            tstart = t[:1] - 0.01
        if tend is None:
            tend = t[-1:] + 0.01
        self.tstart = tstart
        self.tend = tend
        self.mmax = np.size(self.tstart)
        baseline = t[-1] - t[0]
        defaults = {
            "i": SineUniform(0, 1),
            "P_eq": Uniform(0, 50),
            "kappa_2": Uniform(-1, 1),
            "kappa_4": Uniform(-1, 1),
            "c": QuadraticLD(),
            "d": QuadraticLD(),
            "lambda": Uniform(ndim=self.nspots, xmin=-np.pi, xmax=np.pi, wrap=True),
            "beta": Uniform(ndim=self.nspots, xmin=-np.pi / 2, xmax=np.pi / 2),
            "alpha": Uniform(ndim=self.nspots, xmin=0, xmax=np.pi / 4),
            "rho": Uniform(ndim=self.nspots, xmin=0, xmax=1),
            "t_max": Uniform(ndim=self.nspots, xmin=t[0], xmax=t[-1]),
            "tau": Uniform(ndim=self.nspots, xmin=0, xmax=baseline),
            "tau_in": Uniform(ndim=self.nspots, xmin=0, xmax=baseline),
            "tau_out": Uniform(ndim=self.nspots, xmin=0, xmax=baseline),
            "U": Uniform(ndim=self.mmax, xmin=0.9, xmax=1.1),
            "B": Uniform(ndim=self.mmax, xmin=0.9, xmax=1.1),
        }
        super(SpotModel, self).__init__(defaults, t, y, dy, priors)
        self.func = macula

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
        theta = np.atleast_2d(theta).astype(t.dtype)
        if theta.shape[1] != self.jmax:
            raise ValueError("Parameter vector with wrong size.")
        yf = self.func(t, theta, self.tstart, self.tend)
        return yf
