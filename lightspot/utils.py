import numpy as np
from scipy.spatial import Delaunay

__all__ = ['triangulate', 'triangle_area', 'polygon_intersection']


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


def add_point(lst, p):
    for q in lst:
        if np.abs(q[0] - p[0]) < 1e-9 and np.abs(q[1] - p[1]) < 1e-9:
            break
    else:
        lst.append(p)


def is_inside(p, poly):
    j = poly.shape[0] - 1
    res = False
    for i in range(poly.shape[0]):
        if (poly[i, 1] > p[1]) != (poly[j, 1] > p[1]) and \
            p[0] < (poly[j, 0] - poly[i, 0]) * (p[1] - poly[i, 1]) \
                / (poly[j, 1] - poly[i, 1]) + poly[i, 0]:
            res = not res
        j = i
        i = i + 1
    return res


def sort_clockwise(points):
    center = np.mean(points, axis=0)
    new_points = sorted(points, key=lambda p: -np.arctan2(p[1] - center[1], p[0] - center[0]))
    return np.array(new_points)
