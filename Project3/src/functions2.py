import cv2
import numpy as np
from matplotlib import pyplot as plt



def detect_and_label_shapes(image: np.ndarray) -> np.ndarray:
    """
    Detects and labels shapes in the given image.

    This function converts the input image to grayscale, applies the Canny edge detector 
    to find edges, and then finds contours in the edged image. For each contour, the 
    function approximates the contour to a simpler shape, determines the shape type 
    (triangle, square, rectangle, pentagon, or circle), draws the contour on the original 
    image, and labels the shape at its centroid.

    Parameters:
    -----------
    image : np.ndarray
        The input image in which shapes are to be detected. It should be a color image 
        in BGR format.

    Returns:
    --------
    np.ndarray
        The output image with detected shapes labeled and contours drawn.

    Notes:
    ------
    - This function uses the OpenCV library for image processing tasks.
    - The function assumes that the input image is a color image in RGB format.
    """



    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for _, contour in enumerate(contours):
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        
        # Determine the shape
        shape = "unidentified"
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        else:
            shape = "circle"
        
        # Draw the contour and label the shape
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        
        if M["m00"] != 0 :
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
       
    return image

# Function to remove non-quadrilateral shapes
def remove_non_quadrilaterals(image: np.ndarray) -> np.ndarray:

    """
    Removes all non-quadrilateral shapes from the given image, retaining only quadrilaterals.

    This function converts the input image to grayscale, applies edge detection to find edges,
    and then finds contours in the edged image. For each contour, the function approximates
    the contour to a simpler shape and checks if it has four vertices. If it is a quadrilateral,
    it is drawn on a mask. The final result is obtained by performing a bitwise AND operation
    between the original image and the mask, keeping only the quadrilateral shapes.

    Parameters:
    -----------
    image : np.ndarray
        The input image from which non-quadrilateral shapes are to be removed. It should be a
        color image in BGR format.

    Returns:
    --------
    np.ndarray
        The output image with only quadrilateral shapes retained and other shapes removed.

    Notes:
    ------
    - This function uses the OpenCV library for image processing tasks.
    - The function assumes that the input image is a color image in RGB format.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply edge detection
    edged = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape[:2], dtype="uint8")

    for _, contour in enumerate(contours):
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the contour has 4 vertices, it's a quadrilateral
        if len(approx) == 4:
            cv2.drawContours(mask, [approx], -1, 255, -1)
    
    # Bitwise AND to keep only quadrilaterals
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result