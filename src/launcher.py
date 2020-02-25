from src import Main
from src.utils import constants

Main.start(constants.PATH_DATA)


# from src.ocr import extractor as ocr
# from src.debug import draw
# import cv2
# import numpy as np
#
#
# path = "../temp/"
# # path = "../chart_classify/training_dataset/2D-Line-Normal/"
#
# img = cv2.imread(path+"3.jpg")
#
# h_text, v_text = ocr.get_data(img)
#
# both = np.vstack((h_text, v_text))
# draw.draw_texts(img, both)
# cv2.imshow("img", img)
# cv2.waitKey()
#
# print(both)



