import os
import numpy as np
import cv2

from PIL import Image
import pytesseract

cur_path = os.getcwd()
img_name = 'receipt.jpg'
path = os.path.join(cur_path, img_name)

img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)

clone = img.copy()

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

remove1 = []
for i, c1 in enumerate(hulls):

    x, y, w, h = cv2.boundingRect(c1)
    r1_start = (x, y)
    r1_end = (x + w, y + h)

    for j, c2 in enumerate(hulls):

        if i == j:
            continue

        x, y, w, h = cv2.boundingRect(c2)
        r2_start = (x, y)
        r2_end = (x + w, y + h)

        if r1_start[0] > r2_start[0] and r1_start[1] > r2_start[1] and r1_end[0] < r2_end[0] and r1_end[1] < r2_end[1]:
            remove1.append(i)

for j, cnt in enumerate(hulls):
    if j in remove1: continue
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 10
    cv2.rectangle(clone, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 255, 0), 1)

cv2.imshow('mser', clone)
cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for j, cnt in enumerate(hulls):
    if j in remove1: continue
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 10
    cv2.rectangle(mask, (x - margin, y - margin), (x + w + margin, y + h + margin), (255, 255, 255), -1)

text_only = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite("receipt_preprocessing.jpg", text_only)
cv2.imshow("text only", text_only)
cv2.waitKey(0)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(Image.open("receipt_preprocessing.jpg"), lang = "kor")
print(text)
text = pytesseract.image_to_string(Image.open("receipt.jpg"), lang = "kor")
print(text)