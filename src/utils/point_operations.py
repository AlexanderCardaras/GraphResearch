import numpy as np
import math
import cv2
from src.stats import averaging


def points_aligned(point1, point2, axis=1, epsilon=3):
    return abs(point1[axis] - point2[axis]) <= epsilon


def find_aligned_corners(corners, axis, epsilon=3):
    lines = []
    for id1 in range(0, len(corners)):
        temp = []
        p1 = corners[id1][0:2]
        p2 = corners[id1][2:4]
        for id2 in range(id1, len(corners)):
            p3 = corners[id2][0:2]
            p4 = corners[id2][2:4]
            if points_aligned(p1, p3, axis, epsilon) or points_aligned(p1, p4, axis, epsilon) or \
                    points_aligned(p2, p3, axis, epsilon) or points_aligned(p2, p4, axis, epsilon):
                temp.append(corners[id2])

        if len(temp) > 1:
            lines.append(np.array(temp))

    return lines


def find_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    res = np.hstack((centroids, corners))
    return res.astype(int)


def find_axis_candidates(groupings, axis):
    candidates = []
    for group in groupings:
        sorted_group = np.array(sorted(group, key=lambda x: x[abs(1-axis)]))
        distances = np.diff(sorted_group[:, abs(1-axis)])
        grouped_distances, grouped_indices = averaging.group_sequence(distances)

        for idx in range(0, len(grouped_distances)):
            grouped_distance = grouped_distances[idx]
            if len(grouped_distance) > 3:

                average = np.mean(grouped_distance)
                average_array = np.array([average for _ in range(0, len(grouped_distance))])
                average_diff = np.sum(np.absolute(grouped_distance-average_array))/len(grouped_distance)
                candidates.append((average_diff, sorted_group))

    return candidates


def create_line(points, axis):
    min_ = np.min(points[:, abs(axis - 1)::2])
    max_ = np.max(points[:, abs(axis - 1)::2])
    temp = np.mean(points[:, axis::2]).astype(int)
    if axis == 0:
        line = (temp, min_, temp, max_)
    else:
        line = (min_, temp, max_, temp)
    return np.array(line)


# def find_axes(corners):
#     vertical_groupings = find_aligned_points(corners, axis=0)
#     vertical_candidates = find_axis_candidates(vertical_groupings, axis=0)
#     vertical_candidates = np.array(sorted(vertical_candidates, key=lambda x: x[0]))[:, 1]
#     x_axis = np.array(create_line(vertical_candidates[0], axis=0))
#
#     horizontal_groupings = find_aligned_points(corners, axis=1)
#     horizontal_candidates = find_axis_candidates(horizontal_groupings, axis=1)
#     horizontal_candidates = np.array(sorted(horizontal_candidates, key=lambda x: x[0]))[:, 1]
#     y_axis = np.array(create_line(horizontal_candidates[0], axis=1))
#
#     return x_axis, y_axis


def get_chart_region(img, x_axis, y_axis):
    x1 = min(np.min(x_axis[0::2]), np.min(y_axis[0::2]))
    y1 = min(np.min(x_axis[1::2]), np.min(y_axis[1::2]))
    x2 = max(np.max(x_axis[0::2]), np.max(y_axis[0::2]))
    y2 = max(np.max(x_axis[1::2]), np.max(y_axis[1::2]))

    w = x2-x1
    h = y2-y1

    return img[y1:y1+h, x1:x1+w]


def distance_between_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_closest_corner(corner, corners):
    min_distance = np.inf
    closest_corner = None

    for c in corners:
        dist = np.min(np.array([
                distance_between_points(corner[0:2], c[0:2]),
                distance_between_points(corner[0:2], c[2:4]),
                distance_between_points(corner[2:4], c[0:2]),
                distance_between_points(corner[2:4], c[2:4])
                ]))

        if dist < min_distance:
            min_distance = dist
            closest_corner = c

    return corner, closest_corner


def find_closest_corners(corners1, corners2):
    pairs = []
    for c1 in corners1:
        pairs.append(find_closest_corner(c1, corners2))

    return np.array(pairs)
