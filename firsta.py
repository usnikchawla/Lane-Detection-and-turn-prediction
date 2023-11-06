"""
This script performs histogram equalization for a serires of images in adaptive_hist_data.
"""

#Importing modeules for the implementing the task.
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt



#Loading all the images from the folder adaptive_hist_data.
images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

#Iterating thorugh all the images one by one
for img in images:
    
    #Getting the rows and columns of the image.
    row,col=img.shape[:2]
    
    #Iterating through each of the channel of the image.
    for a in range(3):
        channel=img[:,:,a]
        
        #h is the histogram array. Here we use 256 bins.
        h=np.zeros((256,))
        cdf=np.zeros((256,))
        
        #calculating the h_matrix for the channel.
        for i in range(row):
            for j in range(col):
                    
                h[channel[i,j]]+=1
        
        #Calculating the cumulative distributive function for the histogram.        
        cdf[0]=h[0]
        for i in range(1,256):
            cdf[i]=h[i]+cdf[i-1]
        cdf=cdf/(row*col)
        
        #Performing histogram equalization for the current channel of the image.
        for i in range(row):
            for j in range(col):
                    
                img[i,j,a]=np.round(cdf[channel[i,j]]*255)
                
    
    #Displaying the image.
    cv2.imshow("frame",img)
    cv2.waitKey(1)




cv2.destroyAllWindows() 