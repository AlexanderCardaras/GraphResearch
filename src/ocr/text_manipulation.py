import numpy as np
from src.stats import averaging


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_numbers(texts):
    numbers = []
    for text, bounds in texts:
        if is_number(text):
            numbers.append((text, bounds))

    return np.array(numbers)


def get_aligned_numbers(texts):
    numbers = get_numbers(texts)

    vertical = {}
    horizontal = {}

    for number in numbers:
        text, bounds = number
        x, y, w, h = bounds
        h = str(y)
        v = str(x + w)

        if h not in horizontal:
            horizontal[h] = [number]
        else:
            horizontal[h].append(number)

        if v not in vertical:
            vertical[v] = [number]
        else:
            vertical[v].append(number)

    return vertical, horizontal


def group_dictionary(dictionary):
    dict_keys = np.array(list(dictionary.keys()))
    _, indices = averaging.group_sequence(dict_keys.astype(float), maxgap=5)

    grouped_elements = []

    for group in indices:
        elements = []
        num_elements = 0
        for i in group:
            k = dict_keys[i]
            for e in dictionary[k]:
                elements.append(e)
            num_elements += len(dictionary[k])

        grouped_elements.append((num_elements, elements))

        grouped_elements = sorted(grouped_elements, key=lambda x: x[0], reverse=True)
    return np.array(grouped_elements)


def is_below(y, texts):
    for text, bounds in texts:
        x2, y2, w2, h2 = bounds
        return y2 > y


def is_left(x, texts):
    for text, bounds in texts:
        x2, y2, w2, h2 = bounds
        return x2 < x


def pair_axes(vertical, horizontal, features):
    features = sorted(features, key=lambda x: (x[1], x[0]))

    axes = []
    x_axes = {}
    y_axes = {}
    for feature in features:
        x, y, w, h = feature
        for _, texts in horizontal:
            if is_below(y, texts):
                x_axis = texts
                signature = str(x_axis[0][1])
                x_axes[signature] = texts
                break

    for feature in features:
        x, y, w, h = feature
        for _, texts in vertical:
            if is_left(x, texts):
                y_axis = texts
                signature = str(y_axis[0][1])
                y_axes[signature] = texts
                break

    ranges = []
    for x_axis in x_axes:
        y = x_axes[x_axis][0][1][1]
        if len(ranges) == 0:
            ranges.append((0, y, x_axis))
        else:
            ranges.append((ranges[len(ranges)-1][1], y, x_axis))

    temp = {}
    for y_axis in y_axes:
        y_ = y_axes[y_axis]
        for text, bounds in y_:
            x,y,w,h = bounds
            for i in range(0, len(ranges)):
                low, high, sig = ranges[i]
                if y >= low and y < high:
                    if i in temp:
                        temp[i].append((text, bounds))
                    else:
                        temp[i] = [(text, bounds)]

    for i in range(0, len(x_axes.keys())):
        x = x_axes[list(x_axes.keys())[i]]
        y = temp[list(temp.keys())[i]]
        axes.append((x,y))

    return axes


def get_axis_bounds(x_axis, y_axis):
    x1 = np.inf
    x2 = -np.inf
    y1 = np.inf
    y2 = -np.inf

    for text, bound in x_axis:
        x,y,w,h = bound
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x+w)
        y2 = max(y2, y+h)

    for text, bound in y_axis:
        x,y,w,h = bound
        x1 = min(x1, x)
        y1 = min(y1, y)
        x2 = max(x2, x+w)
        y2 = max(y2, y+h)

    return x1, y1, x2-x1+1000, y2-y1


def get_axes(texts, features):

    vertical, horizontal = get_aligned_numbers(texts)

    vertical_group = group_dictionary(vertical)
    horizontal_group = group_dictionary(horizontal)

    axes = pair_axes(vertical_group, horizontal_group, features)

    return axes
