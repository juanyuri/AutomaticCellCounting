# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 17:14:47 2022

@author: Louis
Modifications made by Juan Yuri Díaz Sánchez and Cristina Outeiriño Cid
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import math
from findmaxima2d import find_maxima, find_local_maxima
from maxima import find_local_maxima
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from skimage.segmentation import watershed
from sklearn.naive_bayes import GaussianNB



def big(NAME):


    def backgroundRemove(image):
        """Remove background from an image using contour detection and masking.
        
        Args:
            image (np.ndarray): Input image that we will be removing the background
        
        Returns:
            final (np.ndarray): Image with background removed.
            centerX (float): X-coordinate of the center of the main object.
            centerY (float): Y-coordinate of the center of the main object.
            radius (float): Radius of the minimum enclosing circle around the main object.
        """
        # Convert image to grayscale
        imgray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to obtain a binary image
        thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV,35,1)

        # Find contours in the binary image
        contours, heirarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a black mask with the same dimensions as the input image
        black = np.zeros(img.shape[:2], np.uint8)

        # Determine the largest contour
        largest = max(contours,key=cv2.contourArea)
        (centerX, centerY), radius= cv2.minEnclosingCircle(largest)
        
        # Check if largest contour is greater than 20% of the image area
        if cv2.contourArea(largest)>.2*imgray.shape[0]*imgray.shape[1]:

            # Draw the contour on the mask and use it to mask the input image
            cv2.drawContours(black, [largest], 0, 255, -1)
            ret,black=cv2.threshold(black,0,255,cv2.THRESH_BINARY)
            final=cv2.bitwise_and(img,img,mask=black)

        else:

            # Create a circle on the mask using the minimum enclosing circle
            cv2.circle(black,(int(centerX),int(centerY)),int(radius),255,-1)
            final=cv2.bitwise_and(img,img,mask=black)
            
        # TODO: Show final image

        return final, centerX, centerY, radius



    def changeBackgroundColor(image):
        """Change the background color of an image from black to white."""

        # Identify all pixels that are black
        black_pixels = np.where(
            (image[:, :, 0] == 0) & 
            (image[:, :, 1] == 0) & 
            (image[:, :, 2] == 0)
        )
    
        # Set those pixels to white
        image[black_pixels] = [255, 255, 255]

        return image
    


    def watershed2(image, df):
        """Perform watershed segmentation on an image to isolate suspicious colonies.
        
        Parameters:
            image (np.ndarray): Input image in BGR format (OpenCV default).
            df (pd.DataFrame): DataFrame containing contours of suspicious colonies in a column named 'Contours'.

        Returns:
            contours (list): List of contours found after applying the watershed algorithm.
        """
        # Create a mask from the contours of suspicious colonies
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(mask,df['Contours'].tolist(), -1, 255, -1)     
        
        # Apply the mask to the input image to isolate the regions of interest
        newimg=cv2.bitwise_and(image,image,mask=mask)
        newimg[mask==0] = 255

        # cv2.imshow('whatisthis',newimg)
        
        threshTemp = thresh
        threshTemp = cv2.bitwise_and(threshTemp, threshTemp, mask=mask)

        # cv2.imshow('threshtemp',threshTemp)

        # Perform morphological closing on the thresholded image to remove small holes
        kernel = np.ones((3,3),np.uint8)
        close = cv2.morphologyEx(threshTemp, cv2.MORPH_CLOSE, kernel)
        
        # Identify the sure background by dilating the closed image
        bg = cv2.dilate(close, kernel)
        # cv2.imshow('background',bg)
        
        # Identify the sure foreground
        fg = localMinima2(newimg)
        fg = np.uint8(fg)
    
        # Subtract the foreground from the background to find unknown regions
        unknown = cv2.subtract(bg, fg)

        # Generate markers for the watershed algorithm using connected components
        ret, markers = cv2.connectedComponents(fg)
        markers = markers+1
        markers[unknown==255] = 0

        # Convert the thresholded image for the watershed algorithm
        threshTemp[threshTemp==255] = True
        threshTemp[threshTemp==0] = False
        
        markers= watershed(imgray, markers=markers, mask=threshTemp, compactness=.1)

        #show watershed results
        #     imagecopy=image.copy()
        #     imagecopy[markers==-1]=[255,0,255]
        #     cv2.imshow('debugging',imagecopy)

        # Extract contours from the watershed result
        copyyy=img.copy()
        contours=[]
        for label in np.unique(markers):
            if label==0:
                continue
    
            mask = np.zeros(imgray.shape, dtype="uint8")
            mask[markers == label] = 255

           	# Detect contours in the mask and grab the largest one
            cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(copyyy,cnts,0,(0,0,255),1)
            contours.append(cnts[0])

        # cv2.imshow('debuggingcontours',copyyy)
    
        return contours
    
    

    def nothing(x):
        pass



    def localMinima2(temp):
        """Identify local minima in an image to determine sure foreground areas

        Args:
            temp (numpy.ndarray): Input image in BGR format (OpenCV default).

        Returns:
            black (numpy.ndarray): Binary image where local minima are marked as white (255).
        """

        # Transform image to grayscale and invert it
        temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
        temp=255-temp

        # Find local maxima in the inverted image
        localmax=find_local_maxima(temp)
         
        plt.figure(figsize=(60,60))
        
        # Use the maxima to identify and mark local minima.
        y, x, hehe=find_maxima(temp,localmax,3)

        # Mark the local minima with white
        black=np.zeros(temp.shape, np.uint8)
        for i in range(len(y)):
            black[y,x]=255

        # cv2.imshow('sureforeground',black)

        return black
    
    
    def calcInertia(row):
        """Compute the aspect ratio (inertia) of th ellipse that contains a countour.
        
        Args:
            row: A row from a DataFrame containing contours.

        Returns:
            float: The ratio of the major axis to the minor axis of the fitted ellipse
        """
        if len(row[0]) >= 5:
            (x,y), (MA,ma), angle = cv2.fitEllipse(row['Contours'])
            return MA/ma
        return None
    


    def calcCircularity(row):
        """Compute the circularity of a contour. It tells how close the shape of an object is 
        to a perfect circle
          
        Args:
            row: A row from a DataFrame containing contour information.

        Returns:
            float: The circularity of the contour or None if the perimeter is zero.

        Formulas: 
            circularity: 4 * π * (area / perimeter^2)
        """

         # Extract area and perimeter of the contour
        area = row['Area']
        perimeter = cv2.arcLength(row['Contours'], True)
        
        if perimeter==0:
            return None
        
        circularity=4*math.pi*area/perimeter**2

        return circularity
    


    def calcCircularity2(contour):
        """Compute the circularity of a given contour. It tells how close the shape of an object is 
        to a perfect circle
        
        Args:
            contour (np.ndarray): Contour poinsts

        Returns:
            float: The circularity of the contour or None if the perimeter is zero.

        Formulas: 
            circularity: 4 * π * (area / perimeter^2)
        """

        # Calculate area and perimeter of the contour
        area=cv2.contourArea(contour)
        perimeter=cv2.arcLength(contour,True)

        if perimeter==0:
            return 
        
        circularity=4*math.pi*area/perimeter**2
        return circularity
    


    #look into this and what to do if there are three peaks
    # def Otsu(image): 
    #     hist = cv2.calcHist([image],[0],None,[255],[1,255])
    #     hist=hist.ravel()
    #     hist_norm = hist/hist.sum()
    #     Q = hist_norm.cumsum()
    #     bins = np.arange(256)[1:]
    #     fn_min = np.inf
    #     thresh = -1
    #     for i in range(1,254):
    #         p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    #         q1,q2 = Q[i],Q[253]-Q[i] # cum sum of classes
    #         if q1 < 1.e-6 or q2 < 1.e-6:
    #             continue
    #         b1,b2 = np.hsplit(bins,[i]) # weights
    #         # finding means and variances
    #         m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    #         v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
    #         # calculates the minimization function
    #         fn = v1*q1 + v2*q2
    #         if fn < fn_min:
    #             fn_min = fn
    #             thresh = i
    #     if largestDist(hist)<10:
    #         return False
    #     print('ehehe',thresh)
    #     return thresh
    
    # def largestDist(lst):   
    #     distance=-1
    #     prev=-1
    #     for index, value in enumerate(lst):
    #         if value!=0:
    #             if prev==-1:
    #                 prev=index
    #                 continue
    #             if index-prev>distance:
    #                 distance=(index-prev)
    #                 prev=index
    #     return distance



    def calcColor(row, HSV):
        """Calculate the mean hue value of the region inside the contour.

        Args:
            row (pd.Series): A row from a DataFrame containing contour information.
            HSV (np.ndarray): HSV image.

        Returns:
            float: Mean hue value inside the contour.

        """

        # Create a mask for the contour and compute it
        mask=np.zeros(HSV.shape[:2],np.uint8)
        cv2.drawContours(mask,[row['Contours']],0,255,-1)

        # Calculate the mean hue value inside the masked region.
        color= cv2.mean(HSV[:,:,0], mask=mask)[0]

        return color
    


    def calcAreaRatio(row):
        """Calculate the area ratio of the contour to the area of its minimum
        enclosing circle.

        Parameters:
            row (pd.Series): A row from a DataFrame containing contour information.

        Returns:
            float: The ratio of the contour area to the area of the minimum enclosing circle.

        Algorithm:
        3. Compute the ratio of the contour area to the circle area.

        """
        
        area = row['Area']

        # Calculate the minimum enclosing circle for the contour
        center,radius=cv2.minEnclosingCircle(row['Contours'])
        circleArea=radius**2*math.pi

        # Return the ratio of the contour area to the circle area
        return area/circleArea
    


    def calcCenter(row):
        """Calculate the centroid of the contour using image moments.

        Parameters:
            row (pd.Series): A row from a DataFrame containing contour information.

        Returns:
            tuple: The (x, y) coordinates of the contour's centroid or None.

        Algorithm:
        1. Calculate the moments of the contour.
        2. Compute the centroid coordinates using the moments.

        Note:
        - This function assumes that the 'Contours' column is present in the DataFrame.
        """
        M = cv2.moments(row['Contours'])

        # Check if the area (m00) is zero to avoid division by zero
        if M['m00']==0:
            return None
        
        # Calculate the centroid coordinates
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return cx,cy
        
    

    def contour2df(contours):
        """Convert contours to a DataFrame with various calculated properties.

        Parameters:
            contours (list of np.ndarray): List of contours detected in an image.
            img: Original image in BGR format (used for color conversion).

        Returns:
            Area: Area of each contour.
            Inertia: Aspect ratio of the fitted ellipse to each contour.
            Circularity: Circularity of each contour.
            Color: Mean hue value inside each contour.
            Area Ratio: Ratio of contour area to the area of its minimum enclosing circle.
            Lightness: Mean lightness value inside each contour in LAB color space.
        """
        
        
        df2=pd.DataFrame()
        df2['Contours']=contours

        df2['Area']=df2.apply(lambda row: cv2.contourArea(row['Contours']),axis=1)
        df2['Inertia']=df2.apply(lambda row: calcInertia(row),axis=1)
        df2['Area']=pd.to_numeric(df2['Area'])
        df2['Circularity']=df2.apply(lambda row: calcCircularity(row),axis=1)
        
        HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        df2['Color']=df2.apply(lambda row: calcColor(row,HSV),axis=1)
        df2['areaRatio']=df2.apply(lambda row: calcAreaRatio(row),axis=1)
        
        LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        df2['Lightness']=df2.apply(lambda row: calcColor(row,LAB),axis=1)
        
        print("\nContours DataFrame with Calculated Properties:")
        print(df2.head())
        print("\nSummary Statistics:")
        print(df2.describe())

        return df2
    


    # def equalizeContour(img,contour):
    #     mask=np.zeros(img.shape,dtype='uint8')
    #     cv2.drawContours(mask,contour,-1,255)
    # # =============================================================================
    # #     plt.imshow(mask)
    # # =============================================================================
    #     hist=cv2.calcHist([img], [0], mask, [256], [0,256])
    #     plt.hist(hist,255,[0,255])
    #     img=cv2.equalizeHist(img[mask>0])
    #     hist=cv2.calcHist([img], [0], mask, [256], [0,256])
    #     plt.hist(hist,255,[0,255])
    #     return img
    

    
    def optimalParameters(imgray, color, minWidth):
        """
        Performs adaptive thresholding on a grayscale image to find optimal thresholding parameters.

        Args:
            imgray (numpy.ndarray): A grayscale input image.
            color (list): A list of three integer values representing the desired color range.
            minWidth (int): The minimum width of the histogram peak.

        Returns:
            tuple: A tuple containing the optimal thresholding parameters
            (first, second) and the thresholded image.
        """

        # Compute full image histogram and convert it into 1D array
        fullimage=cv2.calcHist([imgray],[0],None,[252],[4,256])
        fullimage=fullimage.ravel()
        
        # Final variables initialization
        first=0
        second=0
        image=[]

        # Intermediate variables initialization
        yhat=[]
        threshlist=[]
        yhat2=[]
        maxList=[]
        startingValue=minWidth//2*2+1
        endingValue=startingValue*3
        prev=0
        prevThresh=[]


        # This loop iterates through different combinations of block sizes,
        # C values used in the algorithm of Adaptive Thresholding

        # Block Size: higher values result in a smoother thresholding. Lower, detailed thresholding.
        for z in range(4,1,-1): # peaks of the histogram: 4,3,2,1
            largest=0
            peakHeight=0
            yhat=0
            peaks=[]
            
            print('z:',z)

            # Block Size in Adaptive Thresholding 
            for i in range(startingValue, endingValue, 4):
                
                for j in range(1,30,2): 
                    # C value: 1, 2, 3, ..., 29

                    thresh=cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, i, j)
                    hist = cv2.calcHist([imgray],[0],thresh,[252],[4,256])
                    hist=hist.ravel()
                    yhat = savgol_filter(hist, 25, 3)
                    right,left,area,maxidx=maxRange(yhat,hist)
                    # not trivial find the best parameters for this
                    peaks,_=find_peaks(yhat,height=yhat[maxidx]*.1,width=5)

                    if len(peaks)<z and len([i for i in peaks if i>color[1] and i<color[2]])>0:
                        area=sum(yhat[color[1]:color[2]])
                        if area>largest or yhat[color[0]]>peakHeight:
                            largest=area
                            peakHeight=yhat[color[0]]
                            first,second=i,j
                            yhat2=yhat
                            threshlist.append((i,j))
                            image=thresh
                            print(len(peaks),i,j,area)
                            
                            
            print('largest: ',largest)
            

            if largest<.4*prev:
                return first,second,prevThresh
            prev=largest
            prevThresh=image

            maxList.append(maxidx)

        print(threshlist)
        return first,second,image
         
            
            
    def maxRange(yhat, hist):
        """Find the range in the histogram where the maximum value occurs, including the range and its sum.
        
        Args:
            yhat: The array of values (e.g., smoothed histogram) where the maximum needs to be found.
            hist: The original histogram values.

        Returns:
            right (int): Right boundary index of the range.
            left (int): Left boundary index of the range.
            sum (int): Sum of the histogram values within the range.
            maxidx (int): Index of the maximum value in yhat.
        """

        # Right and Left boundaries are expanding
        big=True
        small=True

        # Find the index of the maximum value in yhat
        maxidx=np.argmax(yhat)
        right=maxidx
        left=maxidx
        
        # Expand to the right while the values are non-increasing
        while big or small:
            if right+1<len(yhat) and yhat[right+1]<=yhat[right]:
                right+=1
            else:
                big=False
            
            if left-1>-1 and yhat[left-1]<=yhat[left]:
                left-=1
            else:
                small=False

        return right,left,sum(hist[left-1:right]),maxidx    
        
        
    
    # def drawCircles(image,contours,color,thickness):
    #     for i in contours:
    #         (x,y),radius = cv2.minEnclosingCircle(i)
    #         center = (int(x),int(y))
    #         radius = int(radius)
    #         cv2.circle(image,center,radius,color,thickness)
    #     return image
    
    
    
    def detectColony(event, x, y, flags, param):
        """
        Detect and process a colony in the image based on the mouse click event. It uses
        flood fill algorihtm to detect colonies. It also computes area, circularity and contours
        of each colony.

        Parameters:
            event (int): The type of mouse event (e.g., left button down).
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags (int): Any relevant flags passed by OpenCV.
            param (tuple): Additional parameters.

        """

        if event == cv2.EVENT_LBUTTONDOWN:
            finalUp=0
            prevArea=0
            prevContour=0
            prevMask=0
            prevCircularity=0
            areaList=[]
            circularityList=[]
            
            for i in range(0,50):
                # Create a copy of the grayscale image
                imgraycopy=imgray.copy()
                imgraycopy[imgraycopy==255]=254

                # Apply flood fill
                cv2.floodFill(imgraycopy, None, seedPoint=(x,y), newVal=255,loDiff=50,upDiff=i)
                imgraycopy[imgraycopy<255]=0
                
                # Find contours
                contours,hierarchy=cv2.findContours(imgraycopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                contour=contours[0]
                
                # Calculate area and circularity
                area = cv2.contourArea(contour)
                circularity=calcCircularity2(contour)
                areaList.append(area)
                circularityList.append(circularity)
                print(f'detectColony(): Area {area}. Circularity {circularity}')
                if area-prevArea>500:
                    break
                
                # Update previous values
                prevArea=area
                prevContour=contour
                prevMask=imgraycopy
                prevCircularity=circularity

            # Append the final contour to the contour list
            contourList.append(np.array(prevContour))

            # Set minimum box dimensions for iterative thresholding
            _,_,width,height = cv2.boundingRect(prevContour)
            print('wdith: ',width,'height:',height)
            minWidth.append(max(width,height))
            print('mindWidth:',minWidth)
            finalUp=i-1
            
            # Calculate the mean color in the mask area
            color=cv2.mean(imgray,mask=prevMask)[0]
            print('Final upDiff', finalUp)
                    
            #create dataframe with attributes of ground truth
            
    
            # Calculate the histogram
            hist = cv2.calcHist([imgray],[0],prevMask,[252],[4,256])
            hist=hist.ravel()
            first,last=firstLastIndex(hist)
            colorList.append([int(color),first,last])
            
            # Display the updated image
            imcopy=img.copy()
            cv2.drawContours(imcopy,contourList,len(contourList)-1,(0,0,255),1)
            cv2.imshow('img',imcopy)



    # detectColony2 not used.    
    # def detectColony2(event,x,y,flags,param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #         h=hsv[:,:,2]
    #         R,C=len(h),len(h[0])
    #         color=h[y][x]
    #         print(x,y,color)
    #         def dfs(r,c,col):
    #             if h[r][c]>=col-1 and h[r][c]<=col+1:
    #                 print('row',r,'col',c,'color',h[r][c],'prevColor', col)
    #                 col=h[r][c]
    #                 h[r][c]=-5
    #                 if r>=1:
    #                     dfs(r-1,c,col)
    #                 if r+1<R:
    #                     dfs(r+1,c,col)
    #                 if c>=1:
    #                     dfs(r,c-1,col)
    #                 if c+1<C:
    #                     dfs(r,c+1,col)
    #         dfs(y,x,color)
            
    #         h[h==-5]=255
    #         cv2.imshow('binary',h)
    


    def calcDistance(point1,point2):
        """Calculate the Manhattan distance between two points.

        Args:
            point1 (tuple): Coordinates of the first point (x1, y1).
            point2 (tuple): Coordinates of the second point (x2, y2).

        Returns:
            int: The Manhattan distance between the two points.
        """
        return abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])
    


    def firstLastIndex(histogram):
        """Find the first and second-to-last non-zero indices in a histogram.

        Args:
            histogram (list or numpy.ndarray): The histogram data.

        Returns:
            first_index (int): The first non-zero index.
            last_index (int): The second-to-last non-zero index.
        """

        lst = [i for i,e in enumerate(histogram) if e!=0]

        return lst[0], lst[-2]
    

    
    def resizeImg(img):
        """Resize an image to ensure its total pixel count does not exceed 
        a specified maximum.

        Args:
            img: The original image.

        Returns:
            img: The resized image.
        """

        # Pixel Count Calculation
        height,width=img.shape[:2]
        pixels=height*width

        # Establish Maximum pixel count
        maxPixels = 2000000
        ratio = 1

        # Calculate the resizing ratio if the pixel count exceeds the maximum
        if pixels>maxPixels:
            ratio=maxPixels/pixels
            ratio=math.sqrt(ratio)
        # Resize the image using the calculated ratio
        img=cv2.resize(img,None,fx=ratio,fy=ratio)

        return img
    


    ##########################################################################################################################################################
    ##########################################################################################################################################################
    ##########################################################################################################################################################
    ##########################################################################################################################################################
    
    # Step 1. Initial Setup
    # Create a named window for displaying the image, read image from directory and make a copy
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    imgname=NAME
    colorList=[]
    minWidth=[]
    contourList=[]
    img=cv2.imread(imgname)
    img=resizeImg(img)
    originalImg = img.copy()

    




    # Step 2. Image Preprocessing
    # Set a mouse callback function to detect colonies
    # Apply Median Blur to reduce noise
    # Removes the background and updates the grayscale image
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', detectColony)
    
    img=cv2.medianBlur(img, 3)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img',img)
    
    img,midX,midY,radius=backgroundRemove(img)
    imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    






    # Step 3. Waiting for User Input (detect A priori colonies)
    while(1):
        k=cv2.waitKey(1) & 0xFF
        if k==113: # 'q' key for breaking
            break
        elif k==114: # 'r' key to restart
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return big(NAME)
            






    # Step 4. Processing Ground Truth
    groundTruth=contour2df(contourList)
    print('pre loation length',len(groundTruth))
    groundTruth['Location']=groundTruth.apply(lambda row: calcCenter(row),axis=1)
    print('post liocation length',len(groundTruth))
    print('LOCATION: ',groundTruth['Location'])
    






    # Step 5. Thresholding and Contour Detection:
    thresh=np.zeros(imgray.shape,np.uint8)
    for i in range(len(colorList)):
        first,second,threshold=optimalParameters(imgray,colorList[i],minWidth[i])
        thresh=cv2.bitwise_or(thresh, threshold)
    
    print(first,second)
    cv2.imshow("thresh",thresh)

    contours, hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hierarchy=hierarchy[0,:,3]
    #testing out size of contours
    # areas=[cv2.contourArea(x) for x in contours]
    
    





    # Step 6. DataFrame Initialization
    # For each countour, compute some attributes
    # Data Cleaning: ensures that there are no None values and resets the DataFrame index
    df=pd.DataFrame()
    df['Contours']=contours
    df['Area']=df.apply(lambda row: cv2.contourArea(row['Contours']),axis=1)
    df['Inertia']=df.apply(lambda row: calcInertia(row),axis=1)
    df['Area']=pd.to_numeric(df['Area'])
    df['Circularity']=df.apply(lambda row: calcCircularity(row),axis=1)
    largest=df['Contours'][df['Area'].idxmax()]
    df['hierarchy']=hierarchy
    df['areaRatio']=df.apply(lambda row: calcAreaRatio(row),axis=1)
    df['Location']=df.apply(lambda row: calcCenter(row),axis=1)
    HSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    df['Color']=df.apply(lambda row: calcColor(row,HSV),axis=1)
    LAB=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    df['Lightness']=df.apply(lambda row: calcColor(row,LAB),axis=1)
    
    # Data Cleaning
    df = df.replace(to_replace='None', value=np.nan).dropna()
    df=df.reset_index(drop=True)
    print(df.isnull().any())
    

    


    # Step 7. Mapping Ground Truth to Detected Contours:
    # Maps ground truth contours to the nearest detected contours based on their locations
    realIndexList=[]
    #create new groundTruth
    for index,row in groundTruth.iterrows():
        distance=500
        realIndex=-1

        for index1,row1 in df.iterrows():
            if calcDistance(row['Location'],row1['Location'])<distance:
                distance=calcDistance(row['Location'],row1['Location'])
                realIndex=index1
                print(distance,row['Location'],row1['Location'])
        realIndexList.append(realIndex)
    
    groundTruth=df.iloc[realIndexList]
    
        


    
    # Step 8. Manual Pruning and DBSCAN Clustering
    # Prunes contours based on area and applies DBSCAN clustering to 
    # detect outliers and classify contours.

    # Filter out contours with area greather of 60% of the circle and less than 50% of smallest
    df=df[df['Area']<radius**2*math.pi*.6]
    df=df[df['Area']>.5*(min(groundTruth['Area']))]
    df.reset_index(drop=True,inplace=True)

    # Update ground truth to only include contours present after pruning.
    groundTruth=df[df['Location'].isin(groundTruth['Location'].values)]
        
    imgcopy=img.copy()

    # Prepare data for DBSCAN clustering by removing unnecessary columns.
    df_dbscan=df.copy().drop(['Contours','hierarchy','Lightness','Location'],axis=1)
    

    # Normalize the selected features for clustering.
    scaler=MinMaxScaler()
    df_dbscan[['Area','Circularity','Inertia','Color','areaRatio']]=scaler.fit_transform(df_dbscan[['Area','Circularity','Inertia','Color','areaRatio']])
    
    # Nearest neighbour epsilon is determined
    neigh=NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(df_dbscan)
    distances,indices=nbrs.kneighbors(df_dbscan)
    
    # Sort distances and find the knee point (optimal DBSCAN parameters).

    distances=np.sort(distances,axis=0)
    distances=distances[:,1]
    kn = KneeLocator(np.arange(len(distances)), distances, curve='convex', direction='increasing')
    print('knee: ',kn.knee,distances[kn.knee]*1000)
    
    

    df_copy=pd.DataFrame()
    singleIdx=-2
    imgcopy=img.copy()

    # The DBSCAN is used to group the cells in the 
    # Petri dish into different clusters based on their spatial proximity.
    dbscan=DBSCAN(eps=distances[kn.knee],min_samples=int(len(df)*.05)).fit(df_dbscan)
    
    # Plot the results of DBSCAN clustering for debugging purposes.
    color=iter([[0,0,255],[0,255,0],[255,0,0],[0,255,255],[0,0,0],[255,255,255],[125,125,0],[0,125,125]])
    plotColor=iter(['r','g','b','y','black'])
    df_copy=df.copy()
    df_copy['cluster']=dbscan.labels_
    groups=df_copy.groupby('cluster')
    lst=[]
    
    # For each cluster, draw the contours of the cells belonging to each cluster
    # and compute a metric on the mean circularity and the size of each cluster
    for name, group in groups:
        cv2.drawContours(imgcopy,group['Contours'].tolist(),-1,next(color),1)
        lst.append(group['Circularity'].mean()+len(group)/len(df_copy))           
    
    
    # Determine the dominant cluster index based on circularity and cluster size.
    if lst.index(max(lst))-1==-1:
        lst2=lst[1:]
        print(lst2)
        print(lst)
        singleIdx=lst2.index(max(lst2))
    else:
        singleIdx=lst.index(max(lst))-1
    





    # Step 9. Pre-WaterShed 
    # It filters the DataFrame to include only the relevant rows, 
    # updates the groundTruth DataFrame, and trains a GNB classifier 
    # on the color and lightness features of the cells
    
    # Reset DataFrame index
    df_copy.reset_index(drop=True,inplace=True)
    
    # Update the cluster column in the dataframe groundTruth based on the cluster assignments
    #set groundTruth to include the DBSCAN cluster
    groundTruth['cluster']=df_copy[df_copy['Location'].isin(groundTruth['Location'].values)]['cluster']
    
    # Contains only th erows that belongs to the clusters present in groundTruth
    df_single=df_copy[df_copy['cluster'].isin(groundTruth['cluster'])]
    
    
    # Filter df_single to only contain the correct hierarchy:
    common=df_single['hierarchy'].mode().iloc[0].item()
    df_single=df_single[df_single['hierarchy']==common]
    # NOTUSED: notCircular=df_single[df_single['Circularity']<min(groundTruth['Circularity'])-.1]
    df_single=df_single[df_single['Circularity']>=min(groundTruth['Circularity'])-.1]
    
    # Draw contours of df_single on a copy of the input image for visualization.
    testImage=img.copy()
    cv2.drawContours(testImage,df_single['Contours'].tolist(),-1,(255,255,0),1)

    # Train a Gaussian Naive Bayes GNB classifier on the Color and Lightness features
    # Target: cluster column in the DataFrame
    X=df_single[['Color','Lightness']]
    y=df_single['cluster']
    gnb=GaussianNB()
    gnb.fit(X,y)
    
    
    


    # Step 10. WATERSHED algorithm
    imgcopy=img.copy()

    # Apply watershed algorithm to segment regions of interest (ROIs) based on the contours in df_copy.
    # The function returns markers indicating different segments.
    marker=watershed2(img, df_copy[~df_copy.index.isin(df_single.index)])

    # Convert the markers into a dataframe that contains info about segmented ROIs.
    df_final=contour2df(marker)

    # Filter df_final to include only ROIs with area within a certain range and area ratio higher than a threshold.
    df_final=df_final[(df_final['Area']<df_single['Area'].max()*2) & (df_final['Area']>df_single['Area'].min())]
    df_final=df_final[df_final['areaRatio']>df_single['areaRatio'].min()-.3]

    # If any ROIs are present, predict their clusters with GNB classifier.
    if len(df_final):
        df_final['cluster']=gnb.predict(df_final[['Color','Lightness']])
        

    # Visualize the segmented ROIs and their clusters on the original image.
    clusterList=df_single['cluster'].unique()
    clusterSize=[]
    for count,i in enumerate(clusterList):
        if len(df_final):
            cv2.drawContours(originalImg,df_final[df_final['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            cv2.drawContours(originalImg,df_single[df_single['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            clusterSize.append(len(df_final[df_final['cluster']==i])+len(df_single[df_single['cluster']==i]))
            print('cluster',count,':',clusterSize[-1])
        else:
            cv2.drawContours(originalImg,df_single[df_single['cluster']==i]['Contours'].tolist(),-1,(255*count,255,0),1)
            clusterSize.append(len(df_single[df_single['cluster']==i]))
            print('cluster',count,':',clusterSize[-1])
            
    
    # Calculate the total number of colonies detected after watershed segmentation.
    totalColonies=len(df_single)+len(df_final)





    # Step 10. Final Colony Count and Visualization
    print('original: ',len(df_copy[df_copy['cluster']==singleIdx]))
    print('new',len(df_final))

    # Change the background color of the image for better visualization.
    img=changeBackgroundColor(img)
    
     # Display the total number of colonies on the image.
    print('Total Colony Count: '+str(totalColonies), (int(midX)-200,int(midY-radius-50)))
    
    # Wait for a key press and then close all OpenCV windows.
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
    return originalImg,clusterSize
