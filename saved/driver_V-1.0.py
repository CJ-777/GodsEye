import cv2
import pytesseract
import imutils
import numpy as np
import matplotlib.pyplot as plt

originalImg = cv2.imread("assets/car2.jpg",0)
img = originalImg
# img = cv2.resize(originalImg,(620,480))
# img = imutils.resize(originalImg, width=400)

gray = cv2.bilateralFilter(img, 13, 15, 15)
edged = cv2.Canny(gray, 30, 200)
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(originalImg,img,mask=mask)

(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

_, thresh = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

text = pytesseract.image_to_string(thresh, config='--psm 11')
print("Detected license plate Number is:",text)

plt.imshow(thresh, cmap='gray')
plt.show()