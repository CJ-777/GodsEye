###############################################
# AUTHOR : FANTASTIC FOUR                     #
# TITLE  : GOD'S EYE - ALPR                   #
# DATE   : 4/03/2022                          #
###############################################

import cv2
import pytesseract
import imutils
import numpy as np
import matplotlib.pyplot as plt

##### SHARPENING IMAGE #####
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


##### TAKING IMAGE AS INPUT #####
originalImg = cv2.imread("assets/car1.jpg")
originalImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)

h, w, _ = originalImg.shape

plt.imshow(originalImg)
plt.show()

img = imutils.resize(originalImg, width=400)
img = cv2.cvtColor(originalImg, cv2.COLOR_RGB2GRAY)
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.imshow(img, cmap='gray')
plt.show()

gray = cv2.bilateralFilter(img, 13, 15, 15)
edged = cv2.Canny(gray, 30, 200)

plt.imshow(edged)
plt.show()

# CREATING CONTOURS FROM IMAGE
contours=cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
screenCnt = None

for c in contours[-1:0:-1]:
    peri = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    approx = cv2.approxPolyDP(c, 13, True)
    flag=True
    if len(approx) == 4:
            screenCnt = approx
            break

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

plt.imshow(new_image)
plt.show()

# SECLUDING NUMBERPLATE
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

# GETTING TEXT USING OCR
text = pytesseract.image_to_string(cropped, config='--psm 8 --oem 1 tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', lang='eng')
ans=''
for char in text:
    if char.isalnum() or char==' ':
        ans+=char

print("Detected license plate Number is:",ans)

# FINAL CROPPED NUMBERPLATE
plt.imshow(cropped, cmap='gray')
plt.show()
