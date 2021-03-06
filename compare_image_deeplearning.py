import cv2
import numpy as np

from skimage import io, color

total_image = 1
image_min_idx = 0
image_max_idx = 0
image_percent_min = 10000000000
image_percent_max = -100000000000
result = 0
for idx in range(1, total_image+1):

    # img1 = cv2.imread('Targets/target ({}).png'.format(idx), 0)
    img1 = cv2.imread('output.png')
    img1 = np.uint8(color.rgb2gray(img1) * 255)

    print(img1)
    img2 = cv2.imread('Testing/output ({}).png'.format(66), 0)

    print(img2.shape)

    #--- take the absolute difference of the images ---
    res = cv2.absdiff(img1, img2)

    #--- convert the result to integer type ---
    res = res.astype(np.uint8)

    #--- find percentage difference based on number of pixels that are not zero ---
    percentage = (np.count_nonzero(res) * 100)/ res.size

    result = 100 - percentage

    if (result > image_percent_max):
        image_max_idx = idx
        image_percent_max = result

    if (result < image_percent_min):
        image_min_idx = idx
        image_percent_min = result
    

    print("Image similarity percentage for image-{}: {}%".format(66, result))


# print("\nMinimum and Maximum")
# print("Minimum image percentage accuracy {}% on image-{}".format(image_percent_min, image_min_idx))
# print("Maximum image pecentage accuracy {}% on image-{}".format(image_percent_max, image_max_idx))