import os
import numpy as np
import cv2

from PIL import Image
import pytesseract

cur_path = os.getcwd()
img_name = 'receipt7.jpg'
path = os.path.join(cur_path, img_name)

img = cv2.imread(path)
height, width, channel = img.shape

#그레이 스케일
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("receipt_grayscale_7.jpg", gray)

cv2.imshow('BGR2GRAY', gray)
cv2.waitKey(0)

#다운 샘플링
gray_down = cv2.pyrDown(gray)
cv2.imshow('pyrDown', gray_down)
cv2.waitKey(0)

#가우시안 블러 >> 논문에서는 다운샘플링 후 가우시안 블러를 사용하지만
#처리 시 ocr 결과가 더 나빠져 일단 제외함
#gray_blur = cv2.GaussianBlur(gray_down, (5, 5), 0)
#cv2.imshow('GaussianBlur', gray_blur)
#cv2.waitKey(0)

#샤프닝
sharpening_mask2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
gray_sharpening = cv2.filter2D(gray_down, -1, sharpening_mask2)
cv2.imshow('Blur_and_Sharping', gray_sharpening)
cv2.waitKey(0)

#[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
#[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]


gray_blur = cv2.GaussianBlur(gray_sharpening, (5, 5), 0)
cv2.imshow('GaussianBlur', gray_blur)
cv2.waitKey(0)

#샤프닝 한 영상을 가지고 mser
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray_sharpening)

clone = gray_sharpening.copy()

#mser 결과 가지고 clone 영상에 사각형 표시
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

mask = np.zeros((gray_sharpening.shape[0], gray_sharpening.shape[1], 1), dtype=np.uint8)

for j, cnt in enumerate(hulls):
    if j in remove1: continue
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 10
    cv2.rectangle(mask, (x - margin, y - margin), (x + w + margin, y + h + margin), (255, 255, 255), -1)

text_only = cv2.bitwise_and(gray_sharpening, gray_sharpening, mask=mask)
cv2.imwrite("receipt_preprocessing_7.jpg", text_only)
cv2.imshow("text only", text_only)
cv2.waitKey(0)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(Image.open("receipt_preprocessing_7.jpg"), lang = "kor")
print(text)
print("====================================================================")
text = pytesseract.image_to_string(Image.open("receipt_grayscale_7.jpg"), lang = "kor")
print(text)
print("====================================================================")
text = pytesseract.image_to_string(Image.open("receipt7.jpg"), lang = "kor")
print(text)