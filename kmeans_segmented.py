import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans   
    
# Reference: https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/
    
total_image = 200
for idx in range(1, total_image+1):
    
    pic = cv2.imread('./Refined/transmission_refine_{}.png'.format(idx)) #dividing by 255 to bring the pixel values between 0 and 1
    # print(pic)
    
   # Change color to RGB (from BGR) 
    image = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB) 
  
    # plt.imshow(image) 

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) 
    
    # Convert to float type 
    pixel_vals = np.float32(pixel_vals)


    #the below line of code defines the criteria for the algorithm to stop running,  
    #which will happen is 100 iterations are run or the epsilon (which is the required accuracy)  
    #becomes 85% 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85) 
    
    # then perform k-means clustering wit h number of clusters defined as 3 
    #also random centres are initally chosed for k-means clustering 
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
    
    # convert data into 8-bit values 
    centers = np.uint8(centers) 
    segmented_data = centers[labels.flatten()] 
    
    # reshape data into the original image dimensions 
    segmented_image = segmented_data.reshape((image.shape)) 
    
    # plt.imshow(segmented_image)
    output = "./Segmented_Kmeans/output_refined_kmeans({}).png".format(idx)
    cv2.imwrite(output, segmented_image)
