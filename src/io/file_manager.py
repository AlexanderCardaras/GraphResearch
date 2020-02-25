import os
import csv
from src.io import constants
import numpy as np


def clean_folder(path):
    """
    Deletes everything inside the path directory
    :param path: Path to directory
    :return: None
    """
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    for file in files:
        os.remove(file)


def data_to_matrix(data):
    matrix = []
    num_elements = len(data[constants.LEGEND])
    num_data = len(data[constants.DATA][0])
    num_rows = 1 + num_data

    matrix.append(list(np.array(list(data.keys()))[1:-1]))

    for i in range(0, num_rows):
        row = []
        if i < len(data[constants.FIGURE_NAME]):
            row.append(data[constants.FIGURE_NAME][0])
        else:
            row.append("")

        if i < len(data[constants.X_AXIS]):
            row.append(data[constants.X_AXIS][0])
        else:
            row.append("")

        if i < len(data[constants.Y_AXIS]):
            row.append(data[constants.Y_AXIS][0])
        else:
            row.append("")

        if i < len(data[constants.Y_AXIS]):
            for j in range(0, num_elements):
                row.append(data[constants.LEGEND][j])

        if i > 0:
            for j in range(0, num_elements):
                row.append(data[constants.DATA][j][i-1])

        matrix.append(row)

    return matrix


def write_data(data):

    data = {}
    data[constants.PDF_NAME] = "PIER Study Year 2"
    data[constants.FIGURE_NAME] = [""]
    data[constants.X_AXIS] = ["Month Post Rollover"]
    data[constants.Y_AXIS] = ["Mean Change From Rollover Visual Acuity (ETDRS Letters)"]
    data[constants.DATA] = [[(0, 0), (1, -2.7), (2, -2.9), (3, -1.6), (4, -2.5), (5, 1.8)],
                            [(0, 0), (1, 1.6), (2, 2.2), (3, 2.4), (4, 2), (5, 1.8)],
                            [(0, 0), (1, 1.7), (2, 1.6), (3, 2.3), (4, 3.7), (5, -1.9)]]
    data[constants.LEGEND] = ["Sham", "Ranibizumab 0.3 mg", "Ranibizumab 0.5 mg"]

    file_name = data[constants.PDF_NAME] + ".csv"
    matrix = data_to_matrix(data)

    with open(file_name, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)

        for row in matrix:
            writer.writerow(row)

