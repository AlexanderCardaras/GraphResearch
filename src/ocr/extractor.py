import pytesseract
import cv2
import sys
import numpy as np
from src.utils import rectangle_operations
from src.utils import image_manipulation
from src.stats import averaging
from bars.code import image_show
from src.debug import draw
from PIL import Image
import PIL.Image
from src.utils import constants
import os


# def recognize_text(image):
#     txt = pytesseract.image_to_string(image, config='-l eng --oem 1 --psm 6')
#     return txt


def locate_text(thresh):
    """
    Locates the bounding boxes of blobs, ignoring blobs that are hollow / too 'airy' (such as boxes with no fill)
    :param thresh: Thresholded image
    :return: list of bounding boxes for locations with possible text in them
    """

    # Find all contours in the image
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Calculate the areas of each contour
    contour_areas = []
    for idx in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        contour_areas.append((idx, w * h))

    # Remove contours with areas outside 1 standard deviation of the average
    contour_areas = averaging.reject_outliers(np.array(contour_areas), axis=1, sd=1)

    # Find the contours which have characteristics of text
    text_bounds = []
    for idx, area in contour_areas:
        x, y, w, h = cv2.boundingRect(contours[idx])

        # Calculate the white to black ratio of pixels in the contour
        r = float(cv2.countNonZero(thresh[y:y + h, x:x + w])) / (w * h)

        # Accept contours only if they contain a significant amount of black
        # large hollow boxes and scatters of dots, among other things, will be rejected
        if r > 0.25:
            text_bounds.append((x, y, w, h))

    # Merge nested rectangles
    return rectangle_operations.combine_intersecting_rectangles(text_bounds)


def combine_near_text(text_bounds, scalar=2, max_ratio=3):
    new_text_bounds = []

    for text_bound in text_bounds:
        x, y, w, h = text_bound

        t = max(w, h)

        new_w = int(min(w*max_ratio, int(w+(t-w)*scalar)))
        new_h = int(min(h*max_ratio, int(h+(t-h)*scalar)))

        new_x = int(x-(new_w - w)/scalar)
        new_y = int(y-(new_h - h)/scalar)
        new_text_bounds.append((new_x, new_y, new_w, new_h))

    return rectangle_operations.combine_intersecting_rectangles(new_text_bounds)


def tighten_bounding_boxes(individual, combined, buffer=0):
    final = []
    for cr in combined:
        x1, y1, x2, y2 = (None, None, None, None)
        centers = []

        for text_bound in individual:
            if rectangle_operations.rectangles_intersect(text_bound, cr):
                nx1, ny1, nw, nh = text_bound
                nx2 = nx1 + nw
                ny2 = ny1 + nh

                centers.append(((nx1+nx2/2), (ny1+ny2)/2))
                if x1 is None:
                    x1 = nx1
                    x2 = nx2
                    y1 = ny1
                    y2 = ny2
                else:

                    x1 = min(x1, nx1)
                    x2 = max(x2, nx2)
                    y1 = min(y1, ny1)
                    y2 = max(y2, ny2)

        total_x_diff = 0
        total_y_diff = 0
        for xy1 in centers:
            tx1, ty1 = xy1
            min_x_diff = sys.maxsize
            min_y_diff = sys.maxsize
            for xy2 in centers:
                if xy1 != xy2:
                    tx2, ty2 = xy2
                    min_x_diff = min(min_x_diff, abs(tx1-tx2))
                    min_y_diff = min(min_y_diff, abs(ty1-ty2))

            total_x_diff += min_x_diff
            total_y_diff += min_y_diff

        orientation = total_x_diff-total_y_diff
        final.append(((int(x1)-buffer, int(y1)-buffer, int(x2 - x1)+buffer*2, int(y2 - y1)+buffer*2), orientation))

    return np.array(final)
#
#
# def mask_image_by_text_orientation(img, text_bounds_with_orientation):
#     """
#     Tries to determine which text is vertically oriented and which text is horizontally orientated. Text is separated by
#     Orientation and overlaid onto a black image
#     :param img: Ovencv format image containing text with the text_bounds_with_orientation text on it
#     :param text_bounds_with_orientation: [(x, y, w, h), orientation_score]
#     :return: An image with all the horizontal and vertical oriented text overlaid onto black backgrounds
#     (horizontal, vertical)
#     """
#     horizontal_text = np.zeros(img.shape, np.uint8)
#     vertical_text = np.zeros(img.shape, np.uint8)
#
#     for text_bound, orientation in text_bounds_with_orientation:
#         x, y, w, h = text_bound
#         if orientation >= 0:
#             horizontal_text[y:y+h, x:x+w] = img[y:y+h, x:x+w]
#         else:
#             vertical_text[y:y+h, x:x+w] = img[y:y+h, x:x+w]
#
#     return horizontal_text, vertical_text
#
#
# def text_consistency_check(img, tw, th):
#     h, w = img.shape[0:2]
#     return tw*th/(h*w) < .5

#
# def find_text(horizontal_text_image, vertical_text_image):
#     print("ocr.extractor find_text still needs to implement vertical_text_image")
#
#     raw = pytesseract.image_to_data(horizontal_text_image, output_type=pytesseract.Output.DICT,
#                                     config='-l eng --psm 11')
#
#     h_text = []
#     for i in range(0, len(raw["text"])):
#         if text_consistency_check(horizontal_text_image, raw["width"][i], raw["height"][i]):
#             h_text.append(np.array((raw["text"][i], (raw["left"][i], raw["top"][i], raw["width"][i], raw["height"][i]),
#                                  raw["conf"][i])))
#
#     (h, w) = vertical_text_image.shape[:2]
#     # calculate the center of the image
#     center = (w / 2, h / 2)
#
#     angle90 = -90
#     scale = 1.0
#     # Perform the counter clockwise rotation holding at the center
#     # 90 degrees
#     M = cv2.getRotationMatrix2D(center, angle90, scale)
#     vertical_text_image = cv2.warpAffine(vertical_text_image, M, (h, w))
#
#     raw = pytesseract.image_to_data(vertical_text_image, output_type=pytesseract.Output.DICT,
#                                     config='-l eng --psm 11')
#
#     v_text = []
#     for i in range(0, len(raw["text"])):
#         if text_consistency_check(vertical_text_image, raw["width"][i], raw["height"][i]):
#             v_text.append(np.array((raw["text"][i], (raw["left"][i], raw["top"][i], raw["width"][i], raw["height"][i]),
#                                     raw["conf"][i])))
#
#     return h_text

#
# def spell_check_text(texts, dictionary):
#     final_text = []
#     for text in texts:
#         word = text[0]
#         if len(word) > 0:
#             misspelled = dictionary.unknown(word)
#
#             if len(misspelled) > 0:
#                 for x in misspelled:
#                     final_text.append([dictionary.correction(x), text[1], text[2]])
#             else:
#                 final_text.append(word)
#
#     return final_text
#
#
# def upscale(img, scale=2, bordersize=10):
#     bigger = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#
#     border = cv2.copyMakeBorder(
#         bigger,
#         top=bordersize,
#         bottom=bordersize,
#         left=bordersize,
#         right=bordersize,
#         borderType=cv2.BORDER_CONSTANT,
#         value=[255, 255, 255]
#     )
#
#     blur = cv2.GaussianBlur(border, (3, 3), 0)
#
#     return blur


def rotate_texts(img, texts):
    H, W = img.shape[:2]
    new_texts = []
    for text, bound in texts:
        x, y, w, h = bound
        new_bound = (y, H-x-w, h, w)
        new_texts.append((text, new_bound))

    return np.array(new_texts)


def mask_image(img, bounds):

    # mask
    mask = np.full(img.shape, 255, dtype=np.uint8)

    for rect in bounds:
        x, y, w, h = rect
        mask[y:y+h, x:x+w] = img[y:y+h, x:x+w]

    return mask


def get_text_bounds(img):
    # Convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold image to remove noise
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect possible bounding boxes for individual letters
    individual = locate_text(thresh)

    # Combine bounding boxes to create words
    combined = combine_near_text(individual)

    # Predict the orientation of the words and tighten bounding boxes
    bounds = tighten_bounding_boxes(individual, combined, 2)

    horizontal_bounds = bounds[np.where(bounds[:, 1] >= 0)][:, 0]
    vertical_bounds = bounds[np.where(bounds[:, 1] < 0)][:, 0]

    return thresh, horizontal_bounds, vertical_bounds


def get_text(img):
    image_path = constants.PATH_TEMP + "image_to_text.jpg"
    # image_path = constants.PATH_TEMP + "test1.png"
    cv2.imwrite(image_path, img)
    pil_image = PIL.Image.open(image_path).convert("RGB")
    os.remove(image_path)
    output = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, lang='eng')
    data = np.dstack((output["text"], output["left"], output["top"], output["width"], output["height"]))[0]
    data = data[np.where(data[:, 0] != "")]
    output = []
    for text, x, y, w, h in data:
        output.append((text, (int(x), int(y), int(w), int(h))))
    return np.array(output)


def get_data(img):
    """
    Finds all the text on the image along with its bounding box location
    :param img: OpenCV format image
    """

    thresh, horizontal_bounds, vertical_bounds = get_text_bounds(img)

    horizontal_mask = mask_image(img, horizontal_bounds)
    horizontal_text = get_text(horizontal_mask)

    vertical_mask = mask_image(img, vertical_bounds)
    vertical_mask = image_manipulation.rotate(vertical_mask, 90)
    vertical_text = get_text(vertical_mask)
    vertical_text = rotate_texts(img, vertical_text)

    # draw.draw_texts(img, horizontal_text)
    # draw.draw_rects(img, horizontal_bounds, color=(0,255,0))
    # draw.draw_rects(img, horizontal_text[:, 1])
    # draw.draw_texts(img, vertical_text)
    # draw.draw_rects(img, vertical_text[:, 1])

    # cv2.imshow("img", img)
    # cv2.imshow("horizontal_mask", horizontal_mask)
    # cv2.waitKey()

    return horizontal_text, vertical_text

    # text = get_text(img)


