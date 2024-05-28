import cv2
import numpy as np
from matplotlib import pyplot as plt




# 1st part. Defining our imshow function and showing them
def imshow(image , title , size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


trees=cv2.imread('trees.jpeg')
imshow(trees , "Trees")

trees_R= cv2.split(trees)[2]
imshow(trees_R , "R")