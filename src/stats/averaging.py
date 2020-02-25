import numpy as np
from itertools import groupby, cycle


def reject_outliers(data, axis=0, sd=1):
    """
    Removes items from data which are outside specified standard deviation
    :param data: List of items
    :param axis: Axis to perform calculations on
    :param sd: Number of standard deviations out until data starts being rejected
    :return: List of data inside the 1 standard deviation mark of the average
    """
    return data[abs(data[:, axis] - np.mean(data[:, axis])) < sd * np.std(data[:, axis])]


def group_sequence(orig, maxgap=None):
    order_of_indices = np.argsort(orig)
    data = np.sort(orig)

    if maxgap is None:
        maxgap = np.sum(data)/len(data)/4

    groups = [[data[0]]]
    groups_indices = [[order_of_indices[0]]]

    for idx in range(1, len(data)):
        x = data[idx]
        if abs(x - groups[-1][-1]) <= maxgap:
            groups[-1].append(x)
            groups_indices[-1].append(order_of_indices[idx])
        else:
            groups.append([x])
            groups_indices.append([order_of_indices[idx]])

    # print("orig", orig)
    # print("data", data)
    # print("order_of_indices", order_of_indices)
    # print("groups", groups)
    # print("groups_indices", groups_indices)
    return groups, groups_indices


def consecutive(data, step=1):
    return np.split(data, np.where(np.diff(data) != step)[0]+1)

