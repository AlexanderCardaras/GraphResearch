import cv2
from src.ocr import extractor as ocr
from src.pdf import pdf_reader
from src.utils import constants
from src.io import file_manager
from src.bar import extractor as bar
from src.line import extractor as line
from chart_classify import classify
import numpy as np


def handle_image(figure_type):
    """
    Helper function that organizes which functions need to be called given the type of figure we are scanning
    :param figure_type: The figure classification
    :return: A class capable of handling the extraction of data for said figure_type
    """
    if figure_type == constants.FIGURE_TYPE_2D_BAR_NORMAL:
        return bar
    if figure_type == constants.FIGURE_TYPE_2D_LINE_NORMAL:
        return line

    return None


def scan_pdfs(pdfs):
    """
    Opens pdfs, finds images in pdfs, classifies images, extracts data from images, writes data to csv
    :param pdfs: List of paths to each individual pdf file
    :return: None
    """
    # Before we begin make sure temp folder is empty.
    file_manager.clean_folder(constants.PATH_TEMP)

    for pdf in pdfs:

        # dictionary = pdf_reader.get_dictionary(pdf)

        # Get the paths to the images that were read to the temp folder
        image_paths = pdf_reader.get_images_from_pdf(pdf)
        for path in image_paths:

            # Classify image
            figure_type, confidence = classify.classify_image(path)

            # Find which extractor can handle this figure type
            extractor = handle_image(figure_type)

            if extractor is not None:

                # Load the image in opencv format
                img = cv2.imread(path)

                # Find text in the image
                h_text, v_text = ocr.get_data(img)

                # Extract data using the appropriate extractor class
                data = extractor.get_data(img, np.vstack((h_text, v_text)))

                # Write data to csv
                file_manager.write_data(data)

        # Remove all used images temporarily stored in the temp folder
        file_manager.clean_folder(constants.PATH_TEMP)


def start(entry_path):
    """
    Calls all necessary functions to perform data extraction
    :param entry_path: path to pdf files
    :return: None
    """

    # Get paths to each pdf in the entry point
    pdfs = pdf_reader.get_pdfs(entry_path)

    # Scan every pdf
    scan_pdfs(pdfs[0:])
