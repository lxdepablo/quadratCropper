import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def cropImage(imageName):
    # read the image
    image = cv2.imread(imageName)

    # convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur image
    blur = cv2.medianBlur(grayscale, 5)

    #sharpen image
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # threshold image
    thresh = cv2.threshold(sharpen,100,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # perform contour detection
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (h, w) = image.shape[:2]
    min_area = pow(w/4, 2)
    max_area = pow(w, 2)
    for c in cnts:
        area = cv2.contourArea(c)
        # if contour area is within bounds draw bounding rectangle
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            image = image[y:y+h, x:x+w]

    #cv2.imshow('image', image)
    #cv2.imshow('grayscale', grayscale)
    #cv2.imshow('blur', blur)
    #cv2.imshow('sharpen', sharpen)
    #cv2.imshow('thresh', thresh)
    #cv2.imshow('close', close)

    #cv2.waitKey()

    # reset cwd and save cropped images to croppedImages folder
    os.chdir('../croppedImages')
    imageName = ('.').join(imageName.split('.')[:-1])
    imageName = imageName + '.png'
    cv2.imwrite(imageName, image)
    os.chdir('../uncroppedImages')

# set working directory to uncropped images folder
os.chdir('uncroppedImages')
# put filenames into an array
files = os.listdir()

# crop all files in array
for file in files:
    cropImage(file)
