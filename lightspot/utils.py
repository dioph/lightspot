import numpy as np
from _macula import maculamod
from scipy.spatial import Delaunay

__all__ = ['macula', 'triangulate', 'triangle_area', 'polygon_intersection']


def macula(t, theta_star, theta_spot, theta_inst, derivatives=False, temporal=False, tdeltav=False,
           full_output=False, tstart=None, tend=None):
    """Wrapper for macula FORTRAN routine.

    Parameters
    ----------
    theta_star: array_like
        Array of 12 parameters describing star:
        ! ------------------------------------------------------------------------------
        ! Theta_star(j) = Parameter vector for the star's intrinsic parameters
        ! ------------------------------------------------------------------------------
        ! Istar 	= Theta_star(1)		! Inclination of the star [rads]
        ! Peq 		= Theta_star(2)		! Rot'n period of the star's equator [d]
        ! kappa2 	= Theta_star(3)		! Quadratic differential rotation coeff
        ! kappa4 	= Theta_star(4)		! Quartic differential rotation coeff
        ! c1 		= Theta_star(5)		! 1st of four-coeff stellar LD terms
        ! c2 		= Theta_star(6)		! 2nd of four-coeff stellar LD terms
        ! c3 		= Theta_star(7)		! 3rd of four-coeff stellar LD terms
        ! c4 		= Theta_star(8)		! 4th of four-coeff stellar LD terms
        ! d1 		= Theta_star(9)		! 1st of four-coeff spot LD terms
        ! d2	 	= Theta_star(10)	! 2nd of four-coeff spot LD terms
        ! d3	 	= Theta_star(11)	! 3rd of four-coeff spot LD terms
        ! d4	 	= Theta_star(12)	! 4th of four-coeff spot LD terms

    theta_spot: array_like
        Array of spot parameters, shape (8, Nspot)
        ! ------------------------------------------------------------------------------
        ! Theta_spot(j,k) = Parameters of the k^th spot
        ! ------------------------------------------------------------------------------
        ! Lambda0(k) 	= Theta_spot(1,k)	! Longitude of spot at time tref(k)
        ! Phi0(k) 	= Theta_spot(2,k)	! Latitude of spot at time tref(k)
        ! alphamax(k)	= Theta_spot(3,k)	! Angular spot size at time tmax(k)
        ! fspot(k)	= Theta_spot(4,k)	! Spot-to-star flux contrast of spot k
        ! tmax(k)	= Theta_spot(5,k)	! Time at which spot k is largest
        ! life(k)	= Theta_spot(6,k)	! Lifetime of spot k (FWFM) [days]
        ! ingress(k)	= Theta_spot(7,k)	! Ingress duration of spot k [days]
        ! egress(k)	= Theta_spot(8,k)	! Egress duration of spot k  [days]

    theta_inst: array_like
        Nuisance/instrumental parameters for each of 'm' data sets.
        ! ------------------------------------------------------------------------------
        ! Theta_inst(j,m) = Instrumental/nuisance parameters
        ! ------------------------------------------------------------------------------
        ! U(m) 		= Theta_inst(1,m)	! Baseline flux level for m^th data set
        ! B(m) 		= Theta_inst(2,m)	! Blend factor for m^th data set

    derivatives: bool (optional)
        Whether to calculate derivatives.
    temporal: bool (optional)
        Whether to calculate temporal derivatives
    tdeltav: bool (optional)
        Whether to calculate transit depth variations
    full_output: bool (optional)
        If True, then return all output; otherwise just return model flux.
    tstart, tend: array-like (optional)
        Time stamp at start/end of each of 'm' data sets
    """

    if tstart is None:
        tstart = t[0] - .01
    if tend is None:
        tend = t[-1] + .01
    res = maculamod.macula(t, derivatives, temporal, tdeltav, theta_star, theta_spot, theta_inst, tstart, tend)
    if full_output:
        return res
    else:
        return res[0]


def triangulate(poly):
    tri = Delaunay(poly)
    return poly[tri.simplices]


def triangle_area(triang):
    A, B, C = triang
    area = np.abs(A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1])) / 2
    return area


def polygon_intersection(poly1, poly2):
    poly = []
    for p in poly1:
        if is_inside(p, poly2):
            add_point(poly, p)
    for p in poly2:
        if is_inside(p, poly1):
            add_point(poly, p)
    for i in range(poly1.shape[0]):
        if i == poly1.shape[0] - 1:
            nxt = 0
        else:
            nxt = i + 1
        points = get_intersection_points(poly1[i], poly1[nxt], poly2)
        for p in points:
            add_point(poly, p)
    return sort_clockwise(np.array(poly))


def get_intersection_point(l1p1, l1p2, l2p1, l2p2):
    a1 = l1p2[1] - l1p1[1]
    b1 = l1p1[0] - l1p2[0]
    c1 = a1 * l1p1[0] + b1 * l1p1[1]
    a2 = l2p2[1] - l2p1[1]
    b2 = l2p1[0] - l2p2[0]
    c2 = a2 * l2p1[0] + b2 * l2p1[1]
    det = a1 * b2 - a2 * b1
    if np.abs(det) < 1e-9:
        return np.nan
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    online1 = ((min(l1p1[0], l1p2[0]) < x) or np.isclose(min(l1p1[0], l1p2[0]), x)) \
        and ((max(l1p1[0], l1p2[0]) > x) or np.isclose(max(l1p1[0], l1p2[0]), x)) \
        and ((min(l1p1[1], l1p2[1]) < y) or np.isclose(min(l1p1[1], l1p2[1]), y)) \
        and ((max(l1p1[1], l1p2[1]) > y) or np.isclose(max(l1p1[1], l1p2[1]), y))
    online2 = ((min(l2p1[0], l2p2[0]) < x) or np.isclose(min(l2p1[0], l2p2[0]), x)) \
        and ((max(l2p1[0], l2p2[0]) > x) or np.isclose(max(l2p1[0], l2p2[0]), x)) \
        and ((min(l2p1[1], l2p2[1]) < y) or np.isclose(min(l2p1[1], l2p2[1]), y)) \
        and ((max(l2p1[1], l2p2[1]) > y) or np.isclose(max(l2p1[1], l2p2[1]), y))
    if online1 and online2:
        return np.array([x, y])

    return np.nan


def get_intersection_points(p1, p2, poly):
    intersection = []
    for i in range(poly.shape[0]):
        if i == poly.shape[0] - 1:
            nxt = 0
        else:
            nxt = i + 1
        ip = get_intersection_point(p1, p2, poly[i], poly[nxt])
        if np.all(np.isfinite(ip)):
            add_point(intersection, ip)
    return np.array(intersection)


def add_point(l, p):
    for q in l:
        if np.abs(q[0] - p[0]) < 1e-9 and np.abs(q[1] - p[1]) < 1e-9:
            break
    else:
        l.append(p)


def is_inside(p, poly):
    j = poly.shape[0] - 1
    res = False
    for i in range(poly.shape[0]):
        if (poly[i, 1] > p[1]) != (poly[j, 1] > p[1]) \
                and p[0] < (poly[j, 0] - poly[i, 0]) * (p[1] - poly[i, 1]) / (poly[j, 1] - poly[i, 1]) + poly[i, 0]:
            res = not res
        j = i
        i = i + 1
    return res


def sort_clockwise(points):
    center = np.mean(points, axis=0)
    new_points = sorted(points, key=lambda p: -np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(new_points)
