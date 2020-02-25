"""
Code from newnone at: https://github.com/claird/PyPDF4/blob/master/scripts/pdf-image-extractor.py

Extract images from PDFs without resampling or altering.
Adapted from work by Sylvain Pelissier
http://stackoverflow.com/questions/2693820/extract-images-from-pdf-without-resampling-in-python
"""

from __future__ import print_function
import os
import sys
from os.path import abspath, dirname, join
from src.pdf import pdf_compressor
from src.utils import constants, counter
from PIL import Image
from PyPDF4.pdf import PdfFileReader

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from src.pdf import tokenizer

from io import StringIO

from spellchecker import SpellChecker

spell = SpellChecker()

PROJECT_ROOT = abspath(
    join(dirname(__file__), os.pardir)
)
sys.path.append(PROJECT_ROOT)


def write_image(img, data):
    img.write(data)
    img.close()


def get_images(path):
    images = []
    r = PdfFileReader(open(path, "rb"))
    cnt = counter.Counter()

    for i in range(0, r.numPages):
        page = r.getPage(i)

        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject'].getObject()

            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                    data = xObject[obj].getData()

                    if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                        mode = "RGB"
                    else:
                        mode = "P"

                    write_path = constants.PATH_TEMP + str(cnt.iteration)

                    if '/Filter' in xObject[obj]:
                        if xObject[obj]['/Filter'] == '/FlateDecode':
                            img = Image.frombytes(mode, size, data)
                            write_image(img, data)
                            images.append(write_path + obj[1:] + ".png")

                        elif xObject[obj]['/Filter'] == '/DCTDecode':
                            img = open(write_path + ".jpg", "wb")
                            write_image(img, data)
                            images.append(write_path + ".jpg")

                        elif xObject[obj]['/Filter'] == '/JPXDecode':
                            img = open(write_path + obj[1:] + ".jp2", "wb")
                            write_image(img, data)
                            images.append(write_path + obj[1:] + ".jp2")

                        elif xObject[obj]['/Filter'] == '/CCITTFaxDecode':
                            img = open(write_path + obj[1:] + ".tiff", "wb")
                            write_image(img, data)
                            images.append(write_path + obj[1:] + ".tiff")
                    else:
                        img = Image.frombytes(mode, size, data)
                        write_image(img, data)
                        images.append(write_path + obj[1:] + ".png")

                    cnt.iterate()

    return images


def get_images_from_pdf(pdf):
    # print("opening", pdf)
    p = pdf_compressor.CompressPDF(2)
    images = []
    try:
        images = get_images(pdf)
    except Exception:
        # print(pdf, "is corrupt. Attempting repair.")
        p.compress(pdf, constants.PATH_TEMP + pdf[pdf.rfind('/') + 1:])
        try:
            images = get_images( pdf[pdf.rfind('/') + 1:])
            # print("Successfully repaired", pdf)
            os.remove(constants.PATH_TEMP + pdf[pdf.rfind('/') + 1:])
        except Exception:
            # print("Failed repair. Skipping", pdf)
            print(pdf[21:])
            os.remove(constants.PATH_TEMP + pdf[pdf.rfind('/') + 1:])

    return images


def get_pdfs(root):
    """
    Finds all .pdf files nested within the root directory
    :param root: Path to a folder containing .pdf files
    :return: A list of global paths to every .pdf file located in the root folder
    """
    pdfs = []
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if fname.endswith(".pdf"):
                pdfs.append(join(dirName, fname))

    return pdfs


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                  password=password,
                                  caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


def words_to_file(words, path):
    with open(path, "w") as f:
        for w in words:
            f.write(w)


def get_dictionary(pdf):
    text = convert_pdf_to_txt(pdf)
    unique_words = tokenizer.get_word_list(text)
    spell.word_frequency.load_words(list(unique_words))
    return spell
