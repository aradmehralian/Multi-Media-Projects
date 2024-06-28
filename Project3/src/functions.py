import cv2
import numpy as np
from matplotlib import pyplot as plt


# Defining custom functions used in this project.
def imread(path: str) -> np.ndarray:
    """
    Reads an image given its path and returns the converted image from BGR to RGB.

    Args:
        path (str): Path to the image file.
    
    Returns:
        np.ndarray: Image matrix in RGB format.
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def imshow(image: np.ndarray, title: str, size: (int|float) = 10) -> None:
    """
    Shows image given its matrix along with title. Size of the image can be adjusted.

    Args:
        image (np.ndarray): Image matrix.
        title (str): Title of the image.
        size (int|float): Size of the image to be displayed.

    Returns:
        None
    """
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = h/w
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(image)
    plt.title(title.title())
    plt.show()


def plot_histogram(image: np.ndarray, title: str, color: str) -> None:
    """
    Plots histogram of an image.

    Args:
        image (np.ndarray): Image matrix.
        title (str): Title of the histogram.
        color (str): Color of the histogram.

    Returns:
        None
    """
    plt.hist(image.ravel(), bins=256, range=[0, 256], color=color)
    plt.title(title)
    plt.xlim([0, 256])