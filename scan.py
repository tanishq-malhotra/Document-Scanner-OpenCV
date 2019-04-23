import cv2
import numpy as np
import imutils
from transform import four_pts_transform
from skimage.filters import threshold_local

# reading the image
img = cv2.imread('2.jpg')
# calculating the ratio of the image
# ratio will be used when we pass original image to the four_pts_transform
# thats why we multiply it with the contour while passing it to the function
# but if we perform this on a resized image, then we don't want to multiply with the ratio
#ratio = img.shape[0] / 500.0
original = img.copy()
# resizing the image
img = imutils.resize(img,height=500)

# rotating the image
img = imutils.rotate_bound(img,-90)
# converting the image to greyscale
imgG = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# applying the gaussian blur to the image
imgG = cv2.GaussianBlur(imgG,(5,5),0)
# applying the edge detection on the image
edged = cv2.Canny(imgG,75,200)
# finding the contours on the image, so the object with the largest contour and with 4 points is the
# object we want to scan
contour = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contour = imutils.grab_contours(contour)
contour = sorted(contour,key=cv2.contourArea,reverse=True)[:5]

# we find the contour with four edges
for c in contour:
    # this function calculated the perimeter of the contour and takes 2 param 
    # c and other True means the contour is closed edged contour  
    # perimeter or epsilon
    perimeter = cv2.arcLength(c,True)
    # we calculate contour approximation
    # second param specifies the maximum distance from one contour to other contour
    # used this to get a perfect rectangle if in a case their is round paper
    approx = cv2.approxPolyDP(c,0.02*perimeter,True)
    if len(approx) == 4:
        temp = approx
        break

#cv2.drawContours(img,[temp],-1,(0,255,0),2)
# if we pass the original image
finalImage = four_pts_transform(img,temp.reshape(4,2))
# otherwise
#finalImage = four_pts_transform(img,temp.reshape(4,2))
# converting into gray
finalImage = cv2.cvtColor(finalImage,cv2.COLOR_BGR2GRAY)

thresh = threshold_local(finalImage, 11, offset = 10, method = "gaussian")
finalImage = (finalImage > thresh).astype("uint8") * 255
finalImage = cv2.resize(finalImage,(480,640))
cv2.imshow('image',finalImage)
cv2.waitKey(0)
cv2.destroyAllWindows()