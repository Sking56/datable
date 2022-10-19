#Reference: https://towardsdatascience.com/a-table-detection-cell-recognition-and-text-extraction-algorithm-to-convert-tables-to-excel-files-902edcf289ec

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

input_file = r'in/testinput1.png'
img = cv2.imread(input_file, 0)
img.shape

threshold,img_binary=cv2.threshold(img, 125,255,cv2.THRESH_BINARY_INV)
cv2.imwrite('inverted_img.png',img_binary)

plotted = plt.imshow(img_binary, cmap="gray")
plt.show()

kernel_length = np.array(img).shape[1]

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

