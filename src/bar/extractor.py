import cv2
import numpy as np

from src.utils import colors
from src.stats import averaging
from src.utils import image_manipulation as im
from src.utils import point_operations as po
from src.debug import draw
from src.ocr import text_manipulation as tm
import imutils


def remove_thin_lines(img, th_1=250, th_2=255, kernel=3, correction=7):
    """Isolate bars by removing thin lines."""

    # Threshold image
    thresh = cv2.threshold(img, th_1, th_2, cv2.THRESH_BINARY)[1]

    # get rid of thinner lines
    kernel = np.ones((kernel, kernel), np.uint8)
    isolated = cv2.dilate(thresh, kernel, iterations=3)

    if correction > 0:
        kernel = np.ones((correction, correction), np.uint8)
        isolated = cv2.erode(isolated, kernel, iterations=1)
        return cv2.bitwise_not(isolated)

    if correction < 0:
        kernel = np.ones((-correction, -correction), np.uint8)
        isolated = cv2.dilate(isolated, kernel, iterations=1)
        return cv2.bitwise_not(isolated)

    return cv2.bitwise_not(isolated)


def preprocess_image(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1]

    return remove_thin_lines(thresh, correction=11, kernel=7)


# def find_aligned_points(points, axis, epsilon=3):
#     lines = []
#     for id1 in range(0, len(corners)):
#         temp = []
#         p1 = corners[id1][0:2]
#         p2 = corners[id1][2:4]
#         for id2 in range(id1, len(corners)):
#             p3 = corners[id2][0:2]
#             p4 = corners[id2][2:4]
#             if points_aligned(p1, p3, axis, epsilon) or points_aligned(p1, p4, axis, epsilon) or \
#                     points_aligned(p2, p3, axis, epsilon) or points_aligned(p2, p4, axis, epsilon):
#                 temp.append(corners[id2])
#
#         if len(temp) > 1:
#             lines.append(np.array(temp))
#
#     return lines


def organize_contour(contour):
    organized = []
    for point in contour:
        organized.append(point[0])

    return sorted(organized, key=lambda x: (x[0], -x[1]))


def is_near(p1, p2, epsilon):
    return abs(p1[0] - p2[0]) <= epsilon and abs(p1[1] - p2[1]) <= epsilon


def remove_overlapping_points(contour, epsilon):
    cleaned = []
    used = {}
    for p1 in contour:
        p1_name = str(p1[0])+","+str(p1[1])
        if p1_name in used:
            continue

        similar = [p1]
        used[p1_name] = 0

        for p2 in contour:
            p2_name = str(p2[0]) + "," + str(p2[1])

            if p2_name in used:
                continue

            if p1 is not p2:
                if is_near(p1, p2, epsilon):
                    similar.append(p2)
                    used[p2_name] = 0

        similar = np.array(similar)
        cleaned.append((min(similar[:, 0]), min(similar[:, 1])))

    return cleaned


def get_vertical_pairs(contour, epsilon):
    pairs = []

    used = {}
    for p1 in contour:
        p1_name = str(p1[0]) + "," + str(p1[1])
        if p1_name in used:
            continue

        pair = [p1]
        used[p1_name] = 0

        for p2 in contour:
            p2_name = str(p2[0]) + "," + str(p2[1])

            if p2_name in used:
                continue

            if p1 is not p2:
                if abs(p2[0] - p1[0]) <= epsilon:
                    pair.append(p2)
                    used[p2_name] = 0

        if len(pair) >= 2:
            sorted_pair = sorted(pair, key=lambda x: x[1])
            pairs.append(sorted_pair[:2])

    return pairs


def contour_to_bars(img, contour, epsilon=3):
    contour = organize_contour(contour)
    contour = remove_overlapping_points(contour, epsilon)
    vertical_pairs = get_vertical_pairs(contour, epsilon)

    bars = []
    last = None
    for pair in vertical_pairs:
        if last is None:
            last = pair
        else:
            p1, p2 = pair
            lp1, lp2 = last
            bar_x = min(lp1[0], lp2[0])
            bar_y = max(lp1[1], lp2[1])
            bar_w = max(p1[0], p2[0]) - bar_x
            bar_h = min(p1[1], p2[1]) - bar_y
            bar = (bar_x, bar_y, bar_w, bar_h)
            bars.append(bar)
            last = (max(p1[0], p2[0]), max(lp1[1], lp2[1]))

    return bars


def get_bars(img):
    # convert the resized image to grayscale, blur it slightly, and threshold it
    thresh = preprocess_image(img)

    # find contours in the thresholded image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    bars = []

    # loop over the contours
    for c in contours:
        bar_group = contour_to_bars(img, c)
        for bar in bar_group:
            bars.append(bar)

    return bars


def get_data(img, texts):
    """
    Extracts all information pertaining to the given bar chart
    :param img: OpenCV format image
    :return: List of useful data extracted from the chart
    """

    masked = im.remove_regions(img, texts[:, 1])

    bars = get_bars(masked)

    axes = tm.get_axes(texts, bars)

    x_axis, y_axis = axes[0]
    bounds = tm.get_axis_bounds(x_axis, y_axis)
    for text, tb in texts:
        if tb[0] >= bounds[0] and tb[1] >= bounds[1] and tb[1]+tb[3] < bounds[1]+bounds[3]:
            draw.draw_text(img, (text, tb), color=(255, 0, 0))
    draw.draw_rect(img, bounds, color=(255, 0, 0))
    draw.draw_texts(img, x_axis, color=(255, 0, 0))
    draw.draw_texts(img, y_axis, color=(255, 0, 0))

    x_axis, y_axis = axes[1]
    bounds = tm.get_axis_bounds(x_axis, y_axis)
    for text, tb in texts:
        if tb[0] >= bounds[0] and tb[1] >= bounds[1] and tb[1]+tb[3] < bounds[1]+bounds[3]:
            draw.draw_text(img, (text, tb), color=(0, 255, 0))
    draw.draw_rect(img, bounds, color=(0, 255, 0))
    draw.draw_texts(img, x_axis, color=(0, 255, 0))
    draw.draw_texts(img, y_axis, color=(0, 255, 0))

    x_axis, y_axis = axes[2]
    bounds = tm.get_axis_bounds(x_axis, y_axis)
    for text, tb in texts:
        if tb[0] >= bounds[0] and tb[1] >= bounds[1] and tb[1]+tb[3] < bounds[1]+bounds[3]:
            draw.draw_text(img, (text, tb),  color=(0, 0, 255))
    draw.draw_rect(img, bounds, color=(0, 0, 255))
    draw.draw_texts(img, x_axis, color=(0, 0, 255))
    draw.draw_texts(img, y_axis, color=(0, 0, 255))

    cv2.imshow("img", img)
    cv2.waitKey()


    data = []
    return data
