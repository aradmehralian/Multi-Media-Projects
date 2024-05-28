
# %%
import cv2
import numpy as np
from matplotlib import pyplot as plt







# %%
# 1st part. Defining our imshow function and showing them
def imshow(image , title , size = 10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

Original_image=-cv2.imread('Original_image.jpg')
Transformed_image=-cv2.imread('Transformed_image.jpg')
imshow(Original_image , "Original Image")
imshow(Transformed_image , "Transformed image")



# %%
# 2nd Part. Image Enhancement.

image1=cv2.imread('image1.jfif')
imshow(image1 , "image1")
normalized_image1=image1/255

#Linear transfrom
if normalized_image1.dtype != np.uint8:
   normalized_image1 = (255 * (normalized_image1 - np.min(normalized_image1)) / (np.max(normalized_image1) - np.min(normalized_image1))).astype(np.uint8)


trasform1= 1-normalized_image1
imshow(trasform1 , "Linear Transform")


#Piecewize Linear Transform (1)
piecewise_transformed1 = np.piecewise(
    normalized_image1,
    [normalized_image1 < 0.2, (normalized_image1 >= 0.2) & (normalized_image1 < 0.55), normalized_image1 >= 0.55],
    [lambda x: x , lambda x:0 , lambda x:x]
)
imshow(piecewise_transformed1 , "Piecewise transform (1)" )



#Piecewize Linear Transform (2)
piecewise_transformed2 = np.piecewise(
    normalized_image1,
    [normalized_image1 < 0.4, (normalized_image1 >= 0.4) & (normalized_image1 < 0.55), normalized_image1 >= 0.55],
    [lambda x: 0 , lambda x:x , lambda x:1]
)
imshow(piecewise_transformed2 , "Piecewise transform (2)" )



#Piecewize Linear Transform (3)
piecewise_transformed3 = np.piecewise(
    normalized_image1,
    [normalized_image1 < 0.4, (normalized_image1 >= 0.4) & (normalized_image1 < 0.55), normalized_image1 >= 0.55],
    [lambda x: 1.5*x , lambda x:0.3 , lambda x:0.3/0.55*x]
)
imshow(piecewise_transformed3 , "Piecewise transform (3)" )


#2nd power
image_transformed5= normalized_image1^2
imshow(image_transformed5 , "2nd Power Transformation") 


# #2nd root
# image_transformed4= np.sqrt(normalized_image1)
# imshow(image_transformed4 , "2nd Root Transformation") 





# %%


trees=cv2.imread('trees.jpeg')
imshow(trees , "Trees")

trees_B , trees_G , trees_R= cv2.split(trees)

# Plot histograms
plt.figure(figsize=(15, 5))

# Blue channel histogram
plt.subplot(1, 3, 1)
plt.hist(trees_B.ravel(), bins=256, range=[0, 256], color='blue')
plt.title('Blue Channel Histogram')
plt.xlim([0, 256])

# Green channel histogram
plt.subplot(1, 3, 2)
plt.hist(trees_G.ravel(), bins=256, range=[0, 256], color='green')
plt.title('Green Channel Histogram')
plt.xlim([0, 256])

# Red channel histogram
plt.subplot(1, 3, 3)
plt.hist(trees_R.ravel(), bins=256, range=[0, 256], color='red')
plt.title('Red Channel Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()


# %%
#Histogram Equalization
abraham=cv2.imread('abraham.jpg')


def plot_histogram(image, title, color):
    plt.hist(image.ravel(), bins=256, range=[0, 256], color=color)
    plt.title(title)
    plt.xlim([0, 256])

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(abraham, cv2.COLOR_BGR2HSV)

# Apply histogram equalization to the V channel
hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

# Convert the image back to BGR color space
equalized_color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Display the original and equalized images along with their histograms
plt.figure(figsize=(20, 10))

# Original Color Image
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(abraham, cv2.COLOR_BGR2RGB))
plt.title('Original Color Image')
plt.axis('off')

# Equalized Color Image
plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(equalized_color_image, cv2.COLOR_BGR2RGB))
plt.title('Equalized Color Image')
plt.axis('off')

# Histograms for Original Image
plt.subplot(2, 4, 3)
plot_histogram(abraham[:, :, 0], 'Original Blue Channel Histogram', 'blue')

plt.subplot(2, 4, 4)
plot_histogram(abraham[:, :, 1], 'Original Green Channel Histogram', 'green')

plt.subplot(2, 4, 5)
plot_histogram(abraham[:, :, 2], 'Original Red Channel Histogram', 'red')

# Histograms for Equalized Image
plt.subplot(2, 4, 6)
plot_histogram(equalized_color_image[:, :, 0], 'Equalized Blue Channel Histogram', 'blue')

plt.subplot(2, 4, 7)
plot_histogram(equalized_color_image[:, :, 1], 'Equalized Green Channel Histogram', 'green')

plt.subplot(2, 4, 8)
plot_histogram(equalized_color_image[:, :, 2], 'Equalized Red Channel Histogram', 'red')

plt.tight_layout()
plt.show()



