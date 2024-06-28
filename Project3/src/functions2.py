import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_and_label_shapes(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
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
def remove_non_quadrilaterals(image):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply edge detection
    edged = cv2.Canny(gray, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape[:2], dtype="uint8")

    for i, contour in enumerate(contours):
        # Approximate the contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the contour has 4 vertices, it's a quadrilateral
        if len(approx) == 4:
            cv2.drawContours(mask, [approx], -1, 255, -1)
    
    # Bitwise AND to keep only quadrilaterals
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result