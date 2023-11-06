"""
This script Detects the lane in the image and projects the setection.
This script also interpolates lanes if there is no good detction.
This script also finds the curvature for the lanes and predicts in which direction the lane are turning.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

#Creating a video capture object for the video.
cap=cv2.VideoCapture(sys.argv[1])

#These are the vertices for the trapazoidal region we mask out for working with lane.
vertices=np.array([[31,359],[278,231],[390,231],[621,359]])
#These are the verices for getting the top views of the lane. 
destination=np.array(([0,400],[0,0],[400,0],[400,400]))

#Calculating the homography matrix between region of interset and top view image.
h, status = cv2.findHomography(vertices, destination)
#Calculating the homography matrix between top view image and region of interest.
hinv, status = cv2.findHomography(destination, vertices)

#These are the coeffecients for the predicted left and right lane.
left_fit=None
right_fit=None
    

j=0
while(cap.isOpened()):
    
    #Reading the frame
    ret,frame=cap.read()
    
    if(ret==False):
        break

    #Getting the gray image of the frame and resizing the gray image and the frame to 50%.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dim=gray.shape
    column=int(dim[1]*50/100)
    row=int(dim[0]*50/100)
    di=(column,row)
    gray = cv2.resize(gray, di, interpolation = cv2.INTER_AREA)
    frame =  cv2.resize(frame, di, interpolation = cv2.INTER_AREA)
    
    
    #Perforing contrast limited adaptive histogram eqaulization for the gray image to remove oversatuaration.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    im_out = clahe.apply(gray)
    
    #Creating the mask to get the region of interest
    mask=np.zeros(gray.shape)
    cv2.fillPoly(mask,pts=[vertices],color=(255,255,255))
    mask=(mask==255)
    gray=gray*mask

    
    #Warping the roi into an 400x400 image to get the top view of the lane.
    im_out = cv2.warpPerspective(gray, h, (400,400))
    
    #Performing the thresholding to segment of the lanes.
    T , binary_warped=cv2.threshold(im_out,170,255,cv2.THRESH_BINARY)
    
    #This part is only performed once to get intial lanes.
    if(j==0):
        
        #We get the contours for the lanes.
        contours, hierarchy = cv2.findContours(binary_warped,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_plot=[]

        #Filtering out conters to get the counters for the white lines.
        for i in contours:
            if i.shape[0] > 50:
                contours_plot.append(i)
            
        # cv2.drawContours(im_out, contours_plot, -1, (0, 0, 0), 3)
    
        # cv2.imshow("frame1",im_out)
        
        #getting the midpoint for separating the lane.    
        midpoint = np.int(binary_warped.shape[1]/2)
            
        #Creating the lists to get points inside both the lanes
        leftx=[]
        lefty=[]
        rightx=[]
        righty=[]
        i=0

        #We get 40 lines and iterate the line get points inside the countours.
        while(i<400):
            
            #Checking if for current line some points lie inside the left lane.
            #If we get points then we store the coordinated for these in leftx and lefty list.
            index,=np.where(binary_warped[i,:midpoint]==255)
            if(index.size==0):
                i+=10
                continue
            
            index=index[0]
            
            while(index<midpoint):
                result=None
                for contour in contours_plot:
                    result = cv2.pointPolygonTest(contour, (index,i), False)
                    if result!=-1:
                        leftx.append(index)
                        lefty.append(i)
                        break
                if result!=-1:
                    break 
                index+=1
            
            #Checking if for current line some points lie inside the left lane.
            #If we get points then we store the coordinated for these in rightx and righty list.
            index,=np.where(binary_warped[i,midpoint:]==255)
            if(index.size==0):
                i+=10
                continue
            
            index=index[0]
            
            while(index<400):
                result=None
                for contour in contours_plot:
                    result = cv2.pointPolygonTest(contour, (index,i), False)
                    if result!=-1:
                        rightx.append(index)
                        righty.append(i)
                        break
                        
                if result!=-1:
                    break
                index+=1
            
            i+=10
            
        #Converting the list into numpy array.
        leftx=np.array(leftx)
        lefty=np.array(lefty)
        rightx=np.array(rightx)
        righty=np.array(righty)

        #fitting a 2 degree polynomial through the points.
        left_fit= np.polyfit(lefty, leftx, 2)
        right_fit= np.polyfit(righty, rightx, 2)

        #Creating an colored image to plot the lanes.
        out_img = (np.dstack((im_out, im_out, im_out))).astype('uint8')
        
        #Getting the x and y coordinate for the lane to plot them onto the out_img.
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        #creating the array for plotting the lane.
        leftcurve=np.int_(np.vstack((left_fitx,ploty)).T)
        rightcurve=np.int_(np.vstack((right_fitx,ploty)).T)

        #Plotting the line
        cv2.polylines(out_img,[leftcurve],False,(0,255,0),3)
        cv2.polylines(out_img,[rightcurve],False,(0,0,255),3)

        #calculaing the radius for the left and right lane.
        left_curverad = ((1 + (2*left_fit[0]*399 + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*399 + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        #Averaging out the left and right lane radius.
        radius=(left_curverad+right_curverad)/2
        
        #Using the a parameter of x=ay**2 + by + c to get the direction in which the lanes are turning.
        a=np.around(right_fit[0],5)
        
        
        if a>0:
            text="Right Turn"
        elif a==0:
            text="Go Straight"
        else:
            text="Left Turn"
            
        textr=f"Radius of curvature: {radius}"
        
            

        #warping the out_img back onto the frame.
        final = cv2.warpPerspective(out_img, hinv, (640, 360))
        frame = cv2.add(frame, final)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # fontScale
        fontScale = 0.4

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        frame = cv2.putText(frame, text , (50,25), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, textr , (50,40), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        
            
        
        #Showing the image.    
        cv2.imshow("frame",frame)
        cv2.waitKey(30)
            
        j+=1
    
    else:
        #This part of the code predicts new lane from the previously predicticted lane
        
        #Getting all the coordinates where intesity is not zero in binary_warped()
        nonzero = binary_warped.nonzero()
        nonzeroy= np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        #Collecting all the points that lie with the margin of the aalne.
        margin = 10
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        #fitting a 2 degree polynomial of the obtained points to get the lane.
        left_fit_new = np.polyfit(lefty, leftx, 2)
        right_fit_new = np.polyfit(righty, rightx, 2)

        #Creating an colored image to plot the lanes.
        out_img = (np.dstack((im_out, im_out, im_out))).astype('uint8')
        
        #We check the quality of the predicted lanes.
        #If no good lanes are detected we use the previously predicted lane.
        if(np.abs(left_fit_new[0]-right_fit_new[0])<1):
            right_fit=right_fit_new
            left_fit=left_fit_new
        
        #Getting the x and y coordinate for the lane to plot them onto the out_img.
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
            
        #creating the array for plotting the lane.
        leftcurve=np.int_(np.vstack((left_fitx,ploty)).T)
        rightcurve=np.int_(np.vstack((right_fitx,ploty)).T)

        #Plotting the line
        cv2.polylines(out_img,[leftcurve],False,(0,255,0),5)
        cv2.polylines(out_img,[rightcurve],False,(0,0,255),5)


        #calculaing the radius for the left and right lane.
        left_curverad = ((1 + (2*left_fit[0]*399 + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*399 + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        #Averaging out the left and right lane radius.
        radius=(left_curverad+right_curverad)/2
        
        #Using the a parameter of x=ay**2 + by + c to get the direction in which the lanes are turning.
        a=np.around(right_fit[0],5)
        
        #Using the a parameter of x=ay**2 + by + c to get the direction in which the lanes are turning.
        if a>0:
            text="Right Turn"
        elif a==0:
            text="Go Straight"
        else:
            text="Left Turn"
            
        textr=f"Radius of curvature: {radius}"
        
            

        #warping the out_img back onto the frame.
        final = cv2.warpPerspective(out_img, hinv, (640, 360))
        frame = cv2.add(frame, final)
        
        #font style for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # fontScale
        fontScale = 0.4

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        frame = cv2.putText(frame, text , (50,25), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.putText(frame, textr , (50,40), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        
            
        
        #Showing the output. 
        cv2.imshow("frame",frame)
        cv2.waitKey(30)
        
        
        

cap.release()
cv2.destroyAllWindows()
