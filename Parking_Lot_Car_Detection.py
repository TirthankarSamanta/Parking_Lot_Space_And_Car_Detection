import numpy as np
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Reading image 
img = cv2.imread('sample9.jpg')

# Reading same image in another variable and converting to gray scale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Converting image to a binary image (black and white only image). 
ret,thresh = cv2.threshold(gray,127,255,1)

# Detecting shapes in image by selecting region with same colors or intensity. 
contours,h = cv2.findContours(thresh,1,2)

# Searching through every region selected to find the required polygon for detection. 
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    # Checking if the no. of sides of the selected region is 4 to find rectangle present in every free parking space. 
    if len(approx)==4:
         img1 = cv2.drawContours(img,[cnt],0,(0,255,0),-1)
 
bbox, label, conf = cv.detect_common_objects(img1)
# Identifying the objects present in the picture
output_image = draw_bbox(img1, bbox, label, conf=='')
cv2.imshow("output_image",output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
