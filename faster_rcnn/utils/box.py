"""
Functions to deal with coordinates and areas of union, intersection of rectangles.
"""
from typing import List

def get_width_height(
        rectangle: List[float],
        ) -> List[float]:
    """
    Compute the width and height of the rectangle.

    Args:
        rectangle: rectangle [y1, x1, y2, x2]

    Returns:
        TODO
    """
    y1, x1, y2, x2 = rectangle

    height = abs(y2 - y1)
    width = abs(x2 - x1)

    return (height, width)


def get_center(
        rectangle: List[float],
        ) -> List[float]:
    """
    Compute the coordinates of the center of the rectangle.

    Args:
        rectangle: rectangle [y1, x1, y2, x2]

    Returns:
        The coordinates (x_center, y_center) of the center of the rectangle
    """
    y1, x1, y2, x2 = rectangle

    width, height = get_width_height(rectangle)

    if x1 < x2:
        x_center = x1 + int(width / 2)

    else:
        x_center = x2 + int(width / 2)

    if y1 < y2:
        y_center = y1 + int(height / 2)

    else:
        y_center = y2 + int(height / 2)

    return (x_center, y_center)


def union(
        rectangle_1: List[float],
        rectangle_2: List[float]
        ) -> List[float]:
    """
    Compute the coordinates of the smallest rectangle
    containing both rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [y1,x1,y2,x2]
        rectangle_2: rectangle [y1,x1,y2,x2]

    Returns:
        The union of rectangle_1 and rectangle_2 [y, x, y+h, x+w]
    """

    y_union = min(rectangle_1[0], rectangle_2[0])
    x_union = min(rectangle_1[1], rectangle_2[1])
    height_union = max(rectangle_1[2], rectangle_2[2]) - y_union
    width_union = max(rectangle_1[3], rectangle_2[3]) - x_union

    return [
        y_union,
        x_union,
        y_union + height_union,
        x_union + width_union]


def intersection(
        rectangle_1: List[float],
        rectangle_2: List[float]
        ) -> List[float]:
    """
    Compute the coordinates of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [y1,x1,y2,x2]
        rectangle_2: rectangle [y1,x1,y2,x2]

    Returns:
        The intersection of rectangle_1 and rectangle_2 [y, x, y+h, x+w]
    """

    # rectangle_1 and rectangle_2 should be [x1,y1,x2,y2]
    y_intersect = max(rectangle_1[0], rectangle_2[0])
    x_intersect = max(rectangle_1[1], rectangle_2[1])
    height_intersect = min(rectangle_1[2], rectangle_2[2]) - y_intersect
    width_intersect = min(rectangle_1[3], rectangle_2[3]) - x_intersect

    if width_intersect < 0 or height_intersect < 0:
        return [0, 0, 0, 0]

    return [
        y_intersect,
        x_intersect,
        y_intersect + height_intersect,
        x_intersect + width_intersect]


def union_area(
        rectangle_1: List[float],
        rectangle_2: List[float],
        area_intersection: float
        ) -> float:
    """
    Compute the area of the union of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [y1,x1,y2,x2]
        rectangle_2: rectangle [y1,x1,y2,x2]
        area_intersection: the area of the intersection of rectangle_1 and rectangle_2

    Returns:
        The area of the union of rectangle_1 and rectangle_2
    """

    area_1 = (rectangle_1[2] - rectangle_1[0]) * (rectangle_1[3] - rectangle_1[1])
    area_2 = (rectangle_2[2] - rectangle_2[0]) * (rectangle_2[3] - rectangle_2[1])
    area_union = area_1 + area_2 - area_intersection
    return area_union


def intersection_area(
        rectangle_1: List[float],
        rectangle_2: List[float]
        ) -> float:
    """
    Compute the area of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [y1,x1,y2,x2]
        rectangle_2: rectangle [y1,x1,y2,x2]

    Returns:
        The area of the intersection of rectangle_1 and rectangle_2
    """

    y_intersect = max(rectangle_1[0], rectangle_2[0])
    x_intersect = max(rectangle_1[1], rectangle_2[1])
    height_intersect = min(rectangle_1[2], rectangle_2[2]) - y_intersect
    width_intersect = min(rectangle_1[3], rectangle_2[3]) - x_intersect

    if width_intersect < 0 or height_intersect < 0:
        return 0

    return width_intersect * height_intersect


def iou(
        rectangle_1: List[float],
        rectangle_2: List[float]
        ) -> float:
    """
    Compute the area of the ractangles rectangle_1 and rectangle_2.

    Args:
        rectangle_1: rectangle [y1,x1,y2,x2]
        rectangle_2: rectangle [y1,x1,y2,x2]

    Returns:
        The Intersection over Union for rectangle_1 and rectangle_2
    """

    if rectangle_1[0] >= rectangle_1[2] \
            or rectangle_2[1] >= rectangle_2[3] \
            or rectangle_2[0] >= rectangle_2[2] \
            or rectangle_2[1] >= rectangle_2[3]:
        return 0.0

    area_i = intersection_area(rectangle_1, rectangle_2)
    area_u = union_area(rectangle_1, rectangle_2, area_i)

    return float(area_i) / float(area_u + 1e-6)
