import cv2
import numpy as np
from src.stats import averaging as av
import imutils


def rotate(img, degrees):
    return imutils.rotate_bound(img, degrees)


def remove_regions(img, patches, patch_color=(255, 255, 255)):
    """
    Removes regions of pixels from image and replaces them with patch_color
    :param img: OpenCV format image
    :param patches: List of bounds (x, y, w, h) that will be removed
    :param patch_color: The color that will replace the patches
    :return: A new image with the patch regions replaced with patch_color
    """
    patched_image = np.copy(img)

    for rect in patches:
        x, y, w, h = rect
        patched_image[y:y+h, x:x+w] = patch_color

    return patched_image


def make_three_dim(img):
    """ Returns a 3-channel copy of the image given, or the original image if it is already 3-channel """

    # Is the image 2-channel?
    if len(img.shape) == 2:
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        img2 = np.zeros_like(blank_image)
        img2[:, :, 0] = img
        img2[:, :, 1] = img
        img2[:, :, 2] = img
        return img2

    return img


def split_image(img):
    imgs = [img]

    counts = []
    for i in range(0, img.shape[0]):
        counts.append(np.count_nonzero(np.where(img[i] < 255)))

    counts = np.array(counts)
    spots = np.where(counts == np.min(counts))[0]

    groups = av.consecutive(spots)

    midpoints = []
    for i in range(len(groups)):
        midpoints.append(int(np.mean(groups[i])))

    for x in spots:
        cv2.rectangle(img, (0, x), (830, x), (255, 0, 0), 1)

    cv2.imshow("img", img)
    cv2.waitKey()
    return imgs
