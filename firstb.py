"""
This script performs adaptive histogram equalization for a serires of images in adaptive_hist_data.
In this script we segment the image 8x8 section.
We perform adaptive histogram eqaulization for each of these segmnents.

"""

import cv2
import glob
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import sys




def hist(image , size):
    """
    This section perform adaptive histogram equalization for each of the MXN section.

    Args:
        image (Numpy array): This is one of the segments of the image.
        size (int): the size of the segment.
    """

    #Getting each of the channel of the segment.
    for a in range(3):
        channel=image[:,:,a]
        h=np.zeros((256,))
        cdf=np.zeros((256,))
        
        #h is the histogram array. Here we use 256 bins.
        for i in range(size[0]):
            for j in range(size[1]):
                    
                h[channel[i,j]]+=1
        
         #Calculating the cumulative distributive function for the histogram.     
        cdf[0]=h[0]
        for i in range(1,256):
            cdf[i]=h[i]+cdf[i-1]
        cdf=cdf/(size[0]*size[1])

        #Performing Adaptive histogram equalization for the current channel of the segment.
        for i in range(size[0]):
            for j in range(size[1]):
                    
                image[i,j,a]=np.round(cdf[channel[i,j]]*255)
                


M=int(sys.argv[1])
N=int(sys.argv[2])



#Loading all the images from the folder adaptive_hist_data
images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

#iterating through all the images one by one
for img in images:
    
   
    if((M>img.shape[0]) or (N>img.shape[1])):
        print("Enter valid value for M or N and run the script again.")
        break
    #Segmenting the image into MxN sections and iterating through each of them.
    row,col = img.shape[:2]
    row=int(row/M)
    col=int(col/N)
    for i in range(M):
        for j in range(N):
        
            t=img[i*row:(i+1)*row,j*col:(j+1)*col,:]
            hist(t,(row,col))
    
    
    #Displaying the image.
    cv2.imshow("frame",img)
    cv2.waitKey(1)
    


cv2.destroyAllWindows() 
