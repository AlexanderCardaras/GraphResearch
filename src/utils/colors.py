import numpy as np
import cv2
from src.utils import image_manipulation as im


def change_color(img, target_color, new_color, epsilon=5):
    copy = np.copy(img)
    mask = isolate_color(img, target_color, epsilon)

    copy[np.where((mask == [255, 255, 255]).all(axis=2))] = new_color

    return copy


def isolate_color(img, target_color, epsilon=5):

    lower_bound = np.array(target_color - epsilon, dtype="uint16")
    upper_bound = np.array(target_color + epsilon, dtype="uint16")

    # background_mask = image_show.make_three_dim(cv2.bitwise_not(cv2.inRange(img, lower_bound, upper_bound)))
    background_mask = im.make_three_dim(cv2.inRange(img, lower_bound, upper_bound))

    return background_mask


def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)
