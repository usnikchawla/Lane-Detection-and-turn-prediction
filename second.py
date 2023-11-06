"""
In this script we detect lanes and segment them into a contionous and broken lane.
The continous lane has green color and segmented lane has red color.

"""

import cv2
from matplotlib import lines
import numpy as np
import sys

#Creating a video capture object for the video.
cap=cv2.VideoCapture(sys.argv[1])

#Iterating through the video frame by frame.
while(cap.isOpened()):
    
    #Reading the frame.
    ret,frame=cap.read()
   
    
    if(ret==False):
        break
    
    #Convering the image into gray scale.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #These are the vertices for the trapazoidal region we mask out for working with lane.
    vertices=np.array([[20,539],[414,329],[553,329],[959,539]])

   
    #We create a mask to crop out the area of interest.
    mask=np.zeros(gray.shape)
    cv2.fillPoly(mask,pts=[vertices],color=(255,255,255))
    mask=(mask==255)
    gray=gray*mask

    #Performing thresholding to segment out the lanes.
    T , threshImg=cv2.threshold(gray,170,255,cv2.THRESH_BINARY)

    #Getting edges of the lane.
    edges = cv2.Canny(threshImg,100,200)

    #Getting the lines for the edges.
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, None, 10, 5)

    linesP=np.squeeze(linesP)
    
    #Creating lists to segment out the lines for broken and contionous lane.
    left=[]
    leftdist=0
    right=[]
    rightdist=0
    for line in linesP:

        #Calculating the dist and slope for each of the line.
        dist=np.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)
        slope=(line[1]-line[3])/(line[0]-line[2])
        
        #segmenting the line into two groups.
        if(slope<0):
            left.append(line)
            leftdist+=dist
        else:
            right.append(line)
            rightdist+=dist
            

    
    #Distance for all the lines that corresponds to continous line will be more than that for broken line.
    if(leftdist>rightdist):
        colorleft=(0,255,0)
        colorright=(0,0,255)
        
    else:
        colorleft=(0,0,255)
        colorright=(0,255,0)
        
    #Drwaing the line the image.
    for line in left:
        frame=cv2.line(frame,(line[0],line[1]),(line[2],line[3]),colorleft,3)
    for line in right:
        frame=cv2.line(frame,(line[0],line[1]),(line[2],line[3]),colorright,3)
        

    
    #Displaying the image.
    cv2.imshow("frame",frame)
    cv2.waitKey(60)
    
cap.release()    
cv2.destroyAllWindows()