from src.utils import image_manipulation as im
from src.ocr import extractor as ocr
from src.utils import point_operations as po
from src.debug import draw

import cv2


def find_tick_marks(img):
    x_ticks = []
    y_ticks = []

    im.blob_detection(img)

    return x_ticks, y_ticks


def get_data(img, text):
    # Remove text from the image
    patched_img = im.remove_regions(img, text[:, 1])

    # Find corners
    corners = po.find_corners(patched_img)

    # Find axes
    x_axis, y_axis = po.find_axes(corners)

    # Get chart region
    chart = po.get_chart_region(patched_img, x_axis, y_axis)

    cv2.imshow("line", img)
    cv2.imshow("chart", chart)
    cv2.waitKey()
    lines = []
    return lines
