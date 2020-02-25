
def rectangle_contains_point(rect, point):
    x, y, w, h = rect
    if (x <= point[0]) and (x + w >= point[0]):
        if (y <= point[1]) and (y + h >= point[1]):
            return True

    return False


def rectangles_intersect(rect1, rect2):
    x, y, w, h = rect1

    p1 = (x, y)
    p2 = (x + w, y)
    p3 = (x, y + h)
    p4 = (x + w, y + h)

    if rectangle_contains_point(rect2, p1) or rectangle_contains_point(rect2, p2) or \
            rectangle_contains_point(rect2, p3) or rectangle_contains_point(rect2, p4):
        return True

    return False


def combine_rectangles(rect1, rect2):
    """
    Creates a rectangle that incompasses both rect1 and rect2
    :param rect1: (x, y, w, h)
    :param rect2: (x, y, w, h)
    :return: A rectangle that incompasses both rect1 and rect2
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    new_x = min(x1, x2)
    new_y = min(y1, y2)

    new_w = max(x1+w1, x2+w2) - new_x
    new_h = max(y1+h1, y2+h2) - new_y

    new_rect = (new_x, new_y, new_w, new_h)
    return new_rect


def combine_intersecting_rectangles(rect):
    """
    Combines bounding boxes that are connected or nested within each other
    :param rect: List of (x, y, w, h) tuples
    :return: List of (x, y, w, h) tuples without nesting or connected boundaries
    """

    temp_rect = rect.copy()
    while True:
        found_new_grouping = False
        for rect1 in temp_rect:
            for rect2 in temp_rect:
                if rect1 is not rect2 and rectangles_intersect(rect1, rect2):
                    if rect1 in temp_rect:
                        temp_rect.remove(rect1)

                    temp_rect.remove(rect2)
                    temp_rect.append(combine_rectangles(rect1, rect2))
                    found_new_grouping = True
                    break

        if found_new_grouping is False:
            break

    return temp_rect


def rectangle_averages(rects):
    """
    Computes the average width and height of a list of rectangles
    :param rects: list of rectangles (x, y, w, h)
    :return: average width, average height
    """
    sum_w = 0
    sum_h = 0

    for rect in rects:
        x, y, w, h = rect
        sum_w += w
        sum_h += h

    avg_w = sum_w/len(rects)
    avg_h = sum_h/len(rects)

    return avg_w, avg_h
