from PIL import Image
from scipy import ndimage
import time



def isWithin(x, y, direction, width, height):
    #Depending on where we are and where we are heading, return the appropriate inequality.
    xmax = width - 1
    ymax = height -1
    if direction ==0:
        return (y>0)
    elif direction ==1:
        return (x<xmax and y>0)
    elif direction ==2:
        return (x<xmax)
    elif direction ==3:
        return (x<xmax and y<ymax)
    elif direction ==4:
        return (y<ymax)
    elif direction ==5:
        return (x>0 and y<ymax)
    elif direction ==6:
        return (x>0)
    elif direction ==7:
        return (x>0 and y>0)

    return False;  



def find_local_maxima(img_data):    
    globalMin = np.min(img_data)
    height = img_data.shape[0]
    width = img_data.shape[1]
    dir_x = [0,  1,  1,  1,  0, -1, -1, -1]
    dir_y = [-1, -1,  0,  1,  1,  1,  0, -1]
    out = np.zeros(img_data.shape)
    
    #Goes through each pixel
    for y in range(0,height):
        for x in range(0,width):
            #Reads in the img_data
            v = img_data[y,x]
            #If the pixel is local to the minima of the whole image, can't be maxima.
            if v == globalMin:
                continue
            
            #Is a maxima until proven that it is not.
            isMax = True
            isInner = (y!=0 and y!=height-1) and (x!=0 and x!=width-1)
            for d in range(0,8):
                #Scan each pixel in neighbourhood
                if isInner or isWithin(x,y,d,width,height):
                    #Read the pixels in the neighbourhood.
                    vNeighbour = img_data[y+dir_y[d],x + dir_x[d]]
                    if vNeighbour >v:
                        #We have found there is larger pixel in the neighbourhood.
                        #So this cannot be a local maxima.
                        isMax = False
                        break
            if isMax:
                out[y,x] = 1
    return out




def find_local_maxima_np(img_data):
    #This is the numpy/scipy version of the above function (find local maxima).
    #Its a bit faster, and more compact code.
    
    #Filter data with maximum filter to find maximum filter response in each neighbourhood
    max_out = ndimage.filters.maximum_filter(img_data,size=3)
    #Find local maxima.
    local_max = np.zeros((img_data.shape))
    local_max[max_out == img_data] = 1
    local_max[img_data == np.min(img_data)] = 0
    return local_max