from math import acos, cos, sin, sqrt

from numba import bool_, cuda, float32, float64, int32, void

__all__ = ["cumacula"]

_kernel_cache = {}


def _numba_cumacula_32(
    fmod,
    deltaratio,
    t,
    theta_star,
    theta_spot,
    theta_inst,
    tstart,
    tend,
    c,
    d,
    Fab,
    TdeltaV=False,
):

    F = cuda.grid(1)
    strideF = cuda.gridsize(1)

    # Note - There should be no need to ever change these four parameters.
    pstar = int32(12)
    pspot = int32(8)
    pinst = int32(2)
    pLD = int32(5)
    # VARIABLES
    ndata = int32(t.shape[0])
    Nspot = int32(theta_spot.shape[1])
    mmax = int32(theta_inst.shape[1])
    jmax = pstar + pspot * Nspot + pinst * mmax
    pi = float32(3.1415926)
    halfpi = float32(1.5707963)
    piI = float32(0.31830987)
    zero = float32(0.0)
    one = float32(1.0)
    tol = float32(0.0001)  # alpha values below this will be ignored
    mingress = float32(0.0416667)  # minimum ingress/egress time allowed

    #############################################################################
    ##                       SECTION 2: BASIC PARAMETERS                       ##
    #############################################################################

    # inclination substitutions
    SinInc = sin(theta_star[0])
    CosInc = cos(theta_star[0])

    # Fab0
    Fab0 = zero
    for n in range(pLD):
        Fab0 += (n * c[n]) / (n + float32(4.0))
    Fab0 = one - Fab0

    # MASTER LOOP #############################################################
    for i in range(F, ndata, strideF):
        Fab[i] = Fab0
    # Spot params
    for k in range(Nspot):
        # Phi0 & Prot calculation
        Phi0 = theta_spot[1, k]
        SinPhi0 = sin(Phi0)
        CosPhi0 = cos(Phi0)
        Prot = theta_star[1] / (
            one
            - theta_star[2] * SinPhi0 ** float32(2.0)
            - theta_star[3] * SinPhi0 ** float32(4.0)
        )
        # alpha calculation
        alphamax = theta_spot[2, k]
        fspot = theta_spot[3, k]
        tmax = theta_spot[4, k]
        life = theta_spot[5, k]
        ingress = max(mingress, theta_spot[6, k])
        egress = max(mingress, theta_spot[7, k])
        # macula defines the reference time = maximum spot-size time
        # However, one can change the line below to whatever they wish.
        tref = tmax
        # tcrit points = critical instances in the evolution of the spot
        tcrit1 = tmax - float32(0.5) * life - ingress
        tcrit2 = tcrit1 + ingress
        tcrit3 = tcrit2 + life
        tcrit4 = tcrit3 + egress
        for i in range(F, ndata, strideF):
            fmod[i] = zero
            deltaratio[i] = zero
            # alpha, lambda & beta
            # temporal evolution of alpha
            if t[i] < tcrit1 or t[i] > tcrit4:
                alpha = zero
            elif tcrit2 < t[i] < tcrit3:
                alpha = alphamax
            elif tcrit1 <= t[i] <= tcrit2:
                alpha = alphamax * ((t[i] - tcrit1) / ingress)
            else:
                alpha = alphamax * ((tcrit4 - t[i]) / egress)
            sinalpha = sin(alpha)
            cosalpha = cos(alpha)
            # Lambda & beta calculation
            Lambda = theta_spot[0, k] + float32(2.0) * pi * (t[i] - tref) / Prot
            sinLambda = sin(Lambda)
            cosLambda = cos(Lambda)
            cosbeta = CosInc * SinPhi0 + SinInc * CosPhi0 * cosLambda
            if cosbeta >= one:
                beta = zero
            elif cosbeta <= -one:
                beta = pi
            else:
                beta = acos(cosbeta)
            sinbeta = sin(beta)

            #############################################################################
            ##                        SECTION 3: COMPUTING FMOD                        ##
            #############################################################################

            # zetapos and zetaneg
            if beta < -alpha:
                zetapos = one
            elif beta + alpha > halfpi:
                zetapos = zero
            else:
                zetapos = cos(beta + alpha)
            if beta < alpha:
                zetaneg = one
            elif beta - alpha > halfpi:
                zetaneg = zero
            else:
                zetaneg = cos(beta - alpha)
            # Area A
            if alpha > tol:
                if beta > (halfpi + alpha):
                    # Case IV
                    A = zero
                elif beta < (halfpi - alpha):
                    # Case I
                    A = pi * cosbeta * sinalpha * sinalpha
                else:
                    # Case II & III
                    if abs(cosalpha / sinbeta) >= one:
                        Psi = zero
                    else:
                        Psi = sqrt(one - (cosalpha / sinbeta) * (cosalpha / sinbeta))
                    if (-(cosalpha * cosbeta) / (sinalpha * sinbeta)) >= one:
                        Xi = zero
                    elif (-(cosalpha * cosbeta) / (sinalpha * sinbeta)) <= -one:
                        Xi = sinalpha * pi
                    else:
                        Xi = sinalpha * acos(
                            -(cosalpha * cosbeta) / (sinalpha * sinbeta)
                        )
                    if (cosalpha / sinbeta) >= one:
                        A = zero
                    elif (cosalpha / sinbeta) <= -one:
                        A = pi + Xi * cosbeta * sinalpha - Psi * sinbeta * cosalpha
                    else:
                        A = (
                            acos(cosalpha / sinbeta)
                            + Xi * cosbeta * sinalpha
                            - Psi * sinbeta * cosalpha
                        )
            else:
                A = zero
            q = zero
            # Upsilon & w
            for n in range(pLD):
                Upsilon = zetaneg * zetaneg - zetapos * zetapos
                if zetapos == zetaneg:
                    Upsilon += one
                Upsilon = (
                    sqrt(zetaneg ** float32(n + 4)) - sqrt(zetapos ** float32(n + 4))
                ) / Upsilon
                w = (float32(4.0) * (c[n] - d[n] * fspot)) / (n + float32(4.0))
                w *= Upsilon
                # q
                q += (A * piI) * w
            # Fab
            Fab[i] -= q
    for i in range(F, ndata, strideF):
        for m in range(mmax):
            # U and B assignment
            U = theta_inst[0, m]
            B = theta_inst[1, m]
            # Box-car function (labelled as Pi_m in the paper)
            if tstart[m] < t[i] < tend[m]:
                Box = one
            else:
                Box = zero
            fmod[i] += U * Box * (Fab[i] / (Fab0 * B) + (B - 1.0) / B)
            if TdeltaV:
                deltaratio[i] += B * Box
        # delta {obs}/delta
        if TdeltaV:
            deltaratio[i] = (Fab0 / Fab[i]) / deltaratio[i]
        else:
            deltaratio[i] = one


def _numba_cumacula_64(
    fmod,
    deltaratio,
    t,
    theta_star,
    theta_spot,
    theta_inst,
    tstart,
    tend,
    c,
    d,
    Fab,
    TdeltaV=False,
):

    F = cuda.grid(1)
    strideF = cuda.gridsize(1)

    # Note - There should be no need to ever change these four parameters.
    pstar = 12
    pspot = 8
    pinst = 2
    pLD = 5
    # VARIABLES
    ndata = t.shape[0]
    Nspot = theta_spot.shape[1]
    mmax = theta_inst.shape[1]
    jmax = pstar + pspot * Nspot + pinst * mmax
    pi = 3.141592653589793
    tol = 1e-4  # alpha values below this will be ignored
    mingress = 1 / 24  # minimum ingress/egress time allowed

    #############################################################################
    ##                       SECTION 2: BASIC PARAMETERS                       ##
    #############################################################################

    # inclination substitutions
    SinInc = sin(theta_star[0])
    CosInc = cos(theta_star[0])

    # Fab0
    Fab0 = 0.0
    for n in range(pLD):
        Fab0 += (n * c[n]) / (n + 4.0)
    Fab0 = 1.0 - Fab0

    # MASTER LOOP #############################################################
    for i in range(F, ndata, strideF):
        Fab[i] = Fab0
    # Spot params
    for k in range(Nspot):
        # Phi0 & Prot calculation
        Phi0 = theta_spot[1, k]
        SinPhi0 = sin(Phi0)
        CosPhi0 = cos(Phi0)
        Prot = theta_star[1] / (
            1.0 - theta_star[2] * SinPhi0 ** 2 - theta_star[3] * SinPhi0 ** 4
        )
        # alpha calculation
        alphamax = theta_spot[2, k]
        fspot = theta_spot[3, k]
        tmax = theta_spot[4, k]
        life = theta_spot[5, k]
        ingress = max(mingress, theta_spot[6, k])
        egress = max(mingress, theta_spot[7, k])
        # macula defines the reference time = maximum spot-size time
        # However, one can change the line below to whatever they wish.
        tref = tmax
        # tcrit points = critical instances in the evolution of the spot
        tcrit1 = tmax - 0.5 * life - ingress
        tcrit2 = tmax - 0.5 * life
        tcrit3 = tmax + 0.5 * life
        tcrit4 = tmax + 0.5 * life + egress
        for i in range(F, ndata, strideF):
            fmod[i] = 0.0
            deltaratio[i] = 0.0
            # alpha, lambda & beta
            # temporal evolution of alpha
            if t[i] < tcrit1 or t[i] > tcrit4:
                alpha = 0.0
            elif tcrit2 < t[i] < tcrit3:
                alpha = alphamax
            elif tcrit1 <= t[i] <= tcrit2:
                alpha = alphamax * ((t[i] - tcrit1) / ingress)
            else:
                alpha = alphamax * ((tcrit4 - t[i]) / egress)
            sinalpha = sin(alpha)
            cosalpha = cos(alpha)
            # Lambda & beta calculation
            Lambda = theta_spot[0, k] + 2.0 * pi * (t[i] - tref) / Prot
            sinLambda = sin(Lambda)
            cosLambda = cos(Lambda)
            cosbeta = CosInc * SinPhi0 + SinInc * CosPhi0 * cosLambda
            beta = acos(cosbeta)
            sinbeta = sin(beta)

            #############################################################################
            ##                        SECTION 3: COMPUTING FMOD                        ##
            #############################################################################

            # zetapos and zetaneg
            if beta + alpha < 0.0:
                zetapos = 1.0
            elif beta + alpha > pi / 2:
                zetapos = 0.0
            else:
                zetapos = cos(beta + alpha)
            if beta - alpha < 0.0:
                zetaneg = 1.0
            elif beta - alpha > pi / 2:
                zetaneg = 0.0
            else:
                zetaneg = cos(beta - alpha)
            # Area A
            if alpha > tol:
                if beta > (pi / 2 + alpha):
                    # Case IV
                    A = 0.0
                elif beta < (pi / 2 - alpha):
                    # Case I
                    A = pi * cosbeta * sinalpha ** 2
                else:
                    # Case II & III
                    Psi = sqrt(1.0 - (cosalpha / sinbeta) ** 2)
                    Xi = sinalpha * acos(-(cosalpha * cosbeta) / (sinalpha * sinbeta))
                    A = (
                        acos(cosalpha / sinbeta)
                        + Xi * cosbeta * sinalpha
                        - Psi * sinbeta * cosalpha
                    )
            else:
                A = 0.0
            q = 0.0
            # Upsilon & w
            for n in range(pLD):
                Upsilon = zetaneg ** 2 - zetapos ** 2
                if zetapos == zetaneg:
                    Upsilon += 1.0
                Upsilon = (
                    sqrt(zetaneg ** (n + 4)) - sqrt(zetapos ** (n + 4))
                ) / Upsilon
                w = (4.0 * (c[n] - d[n] * fspot)) / (n + 4.0)
                w *= Upsilon
                # q
                q += (A / pi) * w
            # Fab
            Fab[i] -= q
    for i in range(F, ndata, strideF):
        for m in range(mmax):
            # U and B assignment
            U = theta_inst[0, m]
            B = theta_inst[1, m]
            # Box-car function (labelled as Pi_m in the paper)
            if tstart[m] < t[i] < tend[m]:
                Box = 1.0
            else:
                Box = 0.0
            fmod[i] += U * Box * (Fab[i] / (Fab0 * B) + (B - 1.0) / B)
            if TdeltaV:
                deltaratio[i] += B * Box
        # delta {obs}/delta
        if TdeltaV:
            deltaratio[i] = (Fab0 / Fab[i]) / deltaratio[i]
        else:
            deltaratio[i] = 1.0


def _numba_cumacula_signature(ty):
    return void(
        ty[::1],
        ty[::1],
        ty[::1],
        ty[::1],
        ty[:, ::1],
        ty[:, ::1],
        ty[::1],
        ty[::1],
        ty[::1],
        ty[::1],
        ty[::1],
        bool_,
    )


def _cumacula(
    fmod,
    deltaratio,
    t,
    theta_star,
    theta_spot,
    theta_inst,
    tstart,
    tend,
    c,
    d,
    Fab,
    TdeltaV=False,
):
    if fmod.dtype == "float32":
        numba_type = float32
    elif fmod.dtype == "float64":
        numba_type = float64

    if (str(numba_type)) in _kernel_cache:
        kernel = _kernel_cache[(str(numba_type))]
    else:
        sig = _numba_cumacula_signature(numba_type)
        if fmod.dtype == "float32":
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(sig, fastmath=True)(
                _numba_cumacula_32
            )
            print("Registers(32)", kernel._func.get().attrs.regs)
        elif fmod.dtype == "float64":
            kernel = _kernel_cache[(str(numba_type))] = cuda.jit(sig, fastmath=True)(
                _numba_cumacula_64
            )
            print("Registers(64)", kernel._func.get().attrs.regs)

    gpu = cuda.get_current_device()
    numSM = gpu.MULTIPROCESSOR_COUNT
    threadsperblock = (128,)
    blockspergrid = (numSM * 20,)

    kernel[blockspergrid, threadsperblock](
        fmod,
        deltaratio,
        t,
        theta_star,
        theta_spot,
        theta_inst,
        tstart,
        tend,
        c,
        d,
        Fab,
        TdeltaV,
    )
    cuda.synchronize()


def cumacula(t, theta_star, theta_spot, theta_inst, tstart, tend, TdeltaV=False):

    fmod = cuda.device_array_like(t)
    deltaratio = cuda.device_array_like(t)

    # Check input
    assert t.ndim == 1, "t should be 1-D, shape (ndata,)"
    assert theta_star.ndim == 1, "theta_star should be 1-D, shape (12,)"
    assert theta_star.shape[0] == 12, "Wrong number of star params (there should be 12)"
    assert theta_spot.ndim == 2, "theta_spot should be 2-D, shape (8, Nspot)"
    assert theta_spot.shape[0] == 8, "Wrong number of spot params (there should be 8)"
    assert theta_inst.ndim == 2, "theta_inst should be 2-D, shape (2, mmax)"
    assert theta_inst.shape[0] == 2, "Wrong number of inst params (there should be 2)"
    assert tstart.shape[0] == theta_inst.shape[1], "tstart should have shape (mmax,)"
    assert tend.shape[0] == theta_inst.shape[1], "tend should have shape (mmax,)"

    # Allocate Memory
    pLD = 5
    c = cuda.device_array(shape=(pLD,), dtype=theta_star.dtype)
    d = cuda.device_array(shape=(pLD,), dtype=theta_star.dtype)
    Fab = cuda.device_array_like(t)

    # c and d assignment
    for n in range(1, pLD):
        c[n] = theta_star[n + 3]
        d[n] = theta_star[n + 7]
    c[0] = 1.0 - c[1] - c[2] - c[3] - c[4]  # c0
    d[0] = 1.0 - d[1] - d[2] - d[3] - d[4]  # d0

    _cumacula(
        fmod,
        deltaratio,
        t,
        theta_star,
        theta_spot,
        theta_inst,
        tstart,
        tend,
        c,
        d,
        Fab,
        TdeltaV,
    )

    return fmod.copy_to_host(), deltaratio.copy_to_host()
