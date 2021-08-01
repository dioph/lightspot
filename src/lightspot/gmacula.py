from math import acos, cos, sin, sqrt

import cupy as cp
from numba import cuda, float32, float64, int32, void

__all__ = ["macula"]

_kernel_cache = {}


@cuda.jit(device=True)
def zeta_64(x):
    halfpi = 1.5707963267948966
    if x < 0.0:
        return 1.0
    elif x > halfpi:
        return 0.0
    return cos(x)


@cuda.jit(device=True)
def zeta_32(x):
    halfpi = float32(1.5707963)
    if x < float32(0.0):
        return float32(1.0)
    elif x > halfpi:
        return float32(0.0)
    return cos(x)


@cuda.jit(device=True)
def safe_sqrt_64(x):
    if x <= 0.0:
        return 0.0
    return sqrt(x)


@cuda.jit(device=True)
def safe_sqrt_32(x):
    if x <= float32(0.0):
        return float32(0.0)
    return sqrt(x)


@cuda.jit(device=True)
def safe_acos_64(x):
    pi = 3.141592653589793
    if x <= -1.0:
        return pi
    elif x >= 1.0:
        return 0.0
    return acos(x)


@cuda.jit(device=True)
def safe_acos_32(x):
    pi = float32(3.1415926)
    if x <= float32(-1.0):
        return pi
    elif x >= float32(1.0):
        return float32(0.0)
    return acos(x)


@cuda.jit(device=True)
def macula_singlespot_64(t, theta_star, c, d, theta_spot):
    pLD = 5
    pi = 3.141592653589793
    halfpi = 1.5707963267948966
    inv_pi = 0.3183098861837907
    tol = 0.0001
    mingress = 0.0416667
    SinInc = sin(theta_star[0])
    CosInc = cos(theta_star[0])
    Phi0 = theta_spot[1]
    SinPhi0 = sin(Phi0)
    CosPhi0 = cos(Phi0)
    Prot = theta_star[1] / (
        1.0 - theta_star[2] * SinPhi0 ** 2.0 - theta_star[3] * SinPhi0 ** 4.0
    )
    alphamax = theta_spot[2]
    fspot = theta_spot[3]
    tmax = theta_spot[4]
    life = theta_spot[5]
    ingress = max(mingress, theta_spot[6])
    egress = max(mingress, theta_spot[7])
    tcrit1 = tmax - 0.5 * life - ingress
    tcrit2 = tcrit1 + ingress
    tcrit3 = tcrit2 + life
    tcrit4 = tcrit3 + egress
    if t < tcrit1 or t > tcrit4:
        alpha = 0.0
    elif tcrit2 < t < tcrit3:
        alpha = alphamax
    elif tcrit1 <= t <= tcrit2:
        alpha = alphamax * ((t - tcrit1) / ingress)
    else:
        alpha = alphamax * ((tcrit4 - t) / egress)
    sinalpha = sin(alpha)
    cosalpha = cos(alpha)
    Lambda = theta_spot[0] + 2.0 * pi * (t - tmax) / Prot
    cosLambda = cos(Lambda)
    cosbeta = CosInc * SinPhi0 + SinInc * CosPhi0 * cosLambda
    beta = safe_acos_64(cosbeta)
    sinbeta = sin(beta)
    zetapos = zeta_64(beta + alpha)
    zetaneg = zeta_64(beta - alpha)
    if alpha > tol:
        if beta > (halfpi + alpha):
            A = 0.0
        elif beta < (halfpi - alpha):
            A = pi * cosbeta * sinalpha * sinalpha
        else:
            Psi = safe_sqrt_64(1.0 - (cosalpha / sinbeta) * (cosalpha / sinbeta))
            Xi = sinalpha * safe_acos_64(-(cosalpha * cosbeta) / (sinalpha * sinbeta))
            A = (
                safe_acos_64(cosalpha / sinbeta)
                + Xi * cosbeta * sinalpha
                - Psi * sinbeta * cosalpha
            )
    else:
        A = 0.0
    q = 0.0
    Upsilon_den = zetaneg * zetaneg - zetapos * zetapos
    if zetapos == zetaneg:
        Upsilon_den += 1.0
    for n in range(pLD):
        Upsilon = (sqrt(zetaneg ** (n + 4)) - sqrt(zetapos ** (n + 4))) / Upsilon_den
        w = Upsilon * (4.0 * (c[n] - d[n] * fspot)) / (n + 4.0)
        q += (A * inv_pi) * w
    return q


@cuda.jit(device=True)
def macula_singlespot_32(t, theta_star, c, d, theta_spot):
    pLD = int32(5)
    pi = float32(3.1415926)
    halfpi = float32(1.5707963)
    inv_pi = float32(0.31830987)
    zero = float32(0.0)
    one = float32(1.0)
    tol = float32(0.0001)
    mingress = float32(0.0416667)
    SinInc = sin(theta_star[0])
    CosInc = cos(theta_star[0])
    Phi0 = theta_spot[1]
    SinPhi0 = sin(Phi0)
    CosPhi0 = cos(Phi0)
    Prot = theta_star[1] / (
        one
        - theta_star[2] * SinPhi0 ** float32(2.0)
        - theta_star[3] * SinPhi0 ** float32(4.0)
    )
    alphamax = theta_spot[2]
    fspot = theta_spot[3]
    tmax = theta_spot[4]
    life = theta_spot[5]
    ingress = max(mingress, theta_spot[6])
    egress = max(mingress, theta_spot[7])
    tcrit1 = tmax - float32(0.5) * life - ingress
    tcrit2 = tcrit1 + ingress
    tcrit3 = tcrit2 + life
    tcrit4 = tcrit3 + egress
    if t < tcrit1 or t > tcrit4:
        alpha = zero
    elif tcrit2 < t < tcrit3:
        alpha = alphamax
    elif tcrit1 <= t <= tcrit2:
        alpha = alphamax * ((t - tcrit1) / ingress)
    else:
        alpha = alphamax * ((tcrit4 - t) / egress)
    sinalpha = sin(alpha)
    cosalpha = cos(alpha)
    Lambda = theta_spot[0] + float32(2.0) * pi * (t - tmax) / Prot
    cosLambda = cos(Lambda)
    cosbeta = CosInc * SinPhi0 + SinInc * CosPhi0 * cosLambda
    beta = safe_acos_32(cosbeta)
    sinbeta = sin(beta)
    zetapos = zeta_32(beta + alpha)
    zetaneg = zeta_32(beta - alpha)
    if alpha > tol:
        if beta > (halfpi + alpha):
            A = zero
        elif beta < (halfpi - alpha):
            A = pi * cosbeta * sinalpha * sinalpha
        else:
            Psi = safe_sqrt_32(one - (cosalpha / sinbeta) * (cosalpha / sinbeta))
            Xi = sinalpha * safe_acos_32(-(cosalpha * cosbeta) / (sinalpha * sinbeta))
            A = (
                safe_acos_32(cosalpha / sinbeta)
                + Xi * cosbeta * sinalpha
                - Psi * sinbeta * cosalpha
            )
    else:
        A = zero
    q = zero
    Upsilon_den = zetaneg * zetaneg - zetapos * zetapos
    if zetapos == zetaneg:
        Upsilon_den += one
    for n in range(pLD):
        Upsilon = (
            sqrt(zetaneg ** float32(n + 4)) - sqrt(zetapos ** float32(n + 4))
        ) / Upsilon_den
        w = Upsilon * (float32(4.0) * (c[n] - d[n] * fspot)) / (n + float32(4.0))
        q += (A * inv_pi) * w
    return q


def macula_kernel_64(t, theta, tstart, tend, fmod):
    """
    t: (ndata,)
    theta: (ntrials, jmax)
    nspots: int
    tstart: (mmax,)
    tend: (mmax,)

    fmod: (ntrials, ndata)
    """
    x, y = cuda.grid(2)
    dx, dy = cuda.gridsize(2)
    pstar = 12
    pspot = 8
    pinst = 2
    pLD = 5
    ndata = t.shape[0]
    ntrials = theta.shape[0]
    jmax = theta.shape[1]
    mmax = tstart.shape[0]
    nspots = (jmax - mmax * pinst - pstar) // pspot
    theta_star = cuda.local.array(shape=(4,), dtype=float64)
    c = cuda.local.array(shape=(pLD,), dtype=float64)
    d = cuda.local.array(shape=(pLD,), dtype=float64)
    theta_spot = cuda.local.array(shape=(pspot,), dtype=float64)
    for trial in range(x, ntrials, dx):
        for j in range(4):
            theta_star[j] = theta[trial, j]
        c[0] = 1.0
        d[0] = 1.0
        for n in range(1, pLD):
            c[n] = theta[trial, n + 3]
            d[n] = theta[trial, n + 7]
            c[0] -= c[n]
            d[0] -= d[n]
        Fab0 = 1.0
        for n in range(pLD):
            Fab0 -= (n * c[n]) / (n + 4.0)
        for i in range(y, ndata, dy):
            for m in range(mmax):
                if tstart[m] < t[i] < tend[m]:
                    U = theta[trial, pstar + nspots * pspot + m]
                    B = theta[trial, pstar + nspots * pspot + mmax + m]
                    break
                else:
                    U = 0.0
                    B = 1.0
            Fab = Fab0
            for k in range(nspots):
                for j in range(pspot):
                    theta_spot[j] = theta[trial, pstar + j * nspots + k]
                Fab -= macula_singlespot_64(t[i], theta_star, c, d, theta_spot)
            fmod[trial, i] = U * (Fab / (Fab0 * B) + (B - 1.0) / B)


def macula_kernel_32(t, theta, tstart, tend, fmod):
    """
    t: (ndata,)
    theta: (ntrials, jmax)
    nspots: int
    tstart: (mmax,)
    tend: (mmax,)

    fmod: (ntrials, ndata)
    """
    x, y = cuda.grid(2)
    dx, dy = cuda.gridsize(2)
    pstar = int32(12)
    pspot = int32(8)
    pinst = int32(2)
    pLD = int32(5)
    ndata = int32(t.shape[0])
    ntrials = int32(theta.shape[0])
    jmax = int32(theta.shape[1])
    mmax = int32(tstart.shape[0])
    nspots = (jmax - mmax * pinst - pstar) // pspot
    theta_star = cuda.local.array(shape=(4,), dtype=float32)
    c = cuda.local.array(shape=(5,), dtype=float32)
    d = cuda.local.array(shape=(5,), dtype=float32)
    theta_spot = cuda.local.array(shape=(8,), dtype=float32)
    for trial in range(x, ntrials, dx):
        for j in range(4):
            theta_star[j] = theta[trial, j]
        c[0] = float32(1.0)
        d[0] = float32(1.0)
        for n in range(1, pLD):
            c[n] = theta[trial, n + 3]
            d[n] = theta[trial, n + 7]
            c[0] -= c[n]
            d[0] -= d[n]
        Fab0 = float32(1.0)
        for n in range(pLD):
            Fab0 -= (n * c[n]) / (n + float32(4.0))
        for i in range(y, ndata, dy):
            for m in range(mmax):
                if tstart[m] < t[i] < tend[m]:
                    U = theta[trial, pstar + nspots * pspot + m]
                    B = theta[trial, pstar + nspots * pspot + mmax + m]
                    break
                else:
                    U = float32(0.0)
                    B = float32(1.0)
            Fab = Fab0
            for k in range(nspots):
                for j in range(pspot):
                    theta_spot[j] = theta[trial, pstar + j * nspots + k]
                Fab -= macula_singlespot_32(t[i], theta_star, c, d, theta_spot)
            fmod[trial, i] = U * (Fab / (Fab0 * B) + (B - float32(1.0)) / B)


def _macula_kernel_signature(ty):
    return void(ty[::1], ty[:, ::1], ty[::1], ty[::1], ty[:, ::1])


def _macula(t, theta, tstart, tend, fmod):
    if fmod.dtype == "float32":
        numba_type = float32
    elif fmod.dtype == "float64":
        numba_type = float64

    if (str(numba_type)) in _kernel_cache:
        kernel = _kernel_cache[(str(numba_type))]
    else:
        signature = _macula_kernel_signature(numba_type)
        if fmod.dtype == "float32":
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(
                signature, fastmath=True
            )(macula_kernel_32)
            print("Registers(32)", kernel._func.get().attrs.regs)
        elif fmod.dtype == "float64":
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(
                signature, fastmath=True
            )(macula_kernel_64)
            print("Registers(64)", kernel._func.get().attrs.regs)

    gpu = cuda.get_current_device()
    numSM = gpu.MULTIPROCESSOR_COUNT
    threadsperblock = (4, 64)
    blockspergrid = (numSM, 20)

    kernel[blockspergrid, threadsperblock](t, theta, tstart, tend, fmod)
    cuda.synchronize()


def macula(t, theta, tstart, tend):
    fmod = cp.empty(shape=(theta.shape[0], t.shape[0]), dtype=t.dtype)
    # Check input
    assert t.ndim == 1, "t should be 1-D, shape (ndata,)"
    assert theta.ndim == 2, "theta should be 2-D, shape (ntrials, jmax)"
    assert tstart.ndim == 1, "tstart should be 1-D, shape (mmax,)"
    assert tend.ndim == 1, "tend should be 1-D, shape (mmax,)"
    assert (
        tstart.shape[0] == tend.shape[0]
    ), "tstart and tend should both have shape (mmax,)"
    # Check if sizes are coherent
    pstar = 12
    pspot = 8
    pinst = 2
    jmax = theta.shape[1]
    mmax = tstart.shape[0]
    assert (jmax - mmax * pinst - pstar) % pspot == 0, "Invalid number of parameters!"

    _macula(t, theta, tstart, tend, fmod)
    return fmod
