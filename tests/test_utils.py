import numpy as np

from lightspot.utils import polygon_intersection, triangle_area, triangulate


def test_triangulate():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    tri = triangulate(poly)
    assert len(tri) == 2  # Two triangles


def test_triangle_area():
    tri = np.array([[0, 0], [1, 0], [0, 1]])
    assert triangle_area(tri) == 0.5


def test_polygon_intersection():
    poly1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    poly2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
    result = polygon_intersection(poly1, poly2)
    assert result.shape[0] == 4  # Square intersection
