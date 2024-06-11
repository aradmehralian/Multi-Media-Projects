import cv2
import numpy as np
from matplotlib import pyplot as plt


# Defining our own imshow function 
def imshow(image , title , size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(image)
    plt.title(title.title())
    plt.show()

# Defining our own imread function
def imread(path: str) -> any:
    """
    Read an image given its path and return the converted image from BGR to RGB.
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def plot_histogram(image, title, color):
    """
    Plotting histogram of an image
    """
    plt.hist(image.ravel(), bins=256, range=[0, 256], color=color)
    plt.title(title)
    plt.xlim([0, 256])