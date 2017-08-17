
# coding: utf-8

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import numpy
import statistics
import math
import pickle


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #x = [vertices[0][0][0],vertices[0][1][0],vertices[0][2][0],vertices[0][3][0]]
    #y = [vertices[0][0][1],vertices[0][1][1],vertices[0][2][1],vertices[0][3][1]]
    #plt.plot(x, y, 'b--', lw=4)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

import os
imagelist=os.listdir("camera_cal/")

for x in imagelist:
    image = mpimg.imread('camera_cal/'+x) #Open image first 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print('This image is:', type(image), 'with dimensions:', image.shape)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)
        image =cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
    #write_name = 'corners_found'+str(idx)+'.jpg'
    #cv2.imwrite(write_name, img)
        plt.figure()
        plt.imshow(image)


# In[ ]:





# In[2]:


# if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
        
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
#dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
#mtx = dist_pickle["mtx"]
#dist = dist_pickle["dist"]

#images = glob.glob('camera_cal/cal*.jpg')

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

    img = cv2.drawChessboardCorners(img, (8,6), corners, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst,mtx,dist


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    
    return binary_output
    
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
      # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
   # binary_output = np.copy(img) # Remove this line
    return binary_output

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
# Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output




def pipeline(img, s_thresh=(140, 255)):
    #img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    #img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    
    color_select = np.copy(img)
    red_threshold = 130
    green_threshold = 130
    blue_threshold = 10 
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    # Identify pixels below the threshold
    thresholds = (img[:,:,0] < rgb_threshold[0])                 | (img[:,:,1] < rgb_threshold[1])                 | (img[:,:,2] < rgb_threshold[2])
    color_select[thresholds] = [0,0,0]
    #plt.imshow(color_select)
    
    hsv = cv2.cvtColor(color_select, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    ksize = 3 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 250))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 250))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(40, 250))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) ] = 1
    #print(np.shape(combined))
    color_binary = np.dstack(( np.zeros_like(dir_binary), combined, s_binary))
    
    combined_binary = np.zeros_like(combined)
    combined_binary[(s_binary == 1) | (combined == 1) ] = 1
    #gray = cv2.cvtColor(combined_binary, cv2.COLOR_RGB2GRAY)
    return combined_binary

def corners_unwarp(img, mtx, dist):
    # Restore the camera distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    img_size = (img.shape[1], img.shape[0])
    # Source points on the orignal graph 
    src = np.float32([[585,460],[710,460],[1150,720],[150,720]])
    # Destomatopm points choosen on performance 
    dst = np.float32([[240, 0], 
                      [img_size[0]-200, 0], 
                      [img_size[0]-200, img_size[1]], 
                      [200, img_size[1]]])
    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    rev_M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image with OpenCV 
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M, rev_M


image = mpimg.imread('test_images/test5.jpg') 
undistorted,mtx,dist = cal_undistort(image, objpoints, imgpoints)
result = pipeline(undistorted)
top_down, perspective_M,rev_M = corners_unwarp(result, mtx, dist)
#result = fitting_projection(undistorted,top_down,y_axis,lx,rx,rev_M);  
plt.imshow(top_down, cmap='gray')





# In[ ]:





# In[3]:


window_width = 70 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 40 # How much to slide left and right for searching

#warped = top_down


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output



def find_window_centroids(warped, window_width, window_height, margin):
    #import numpy as np
    #histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    #plt.plot(histogram)
    
    #Variables in order to store the x, y information
    lx=[]
    rx=[]
    y_axis =[] 
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    lx.append(l_center)
    rx.append(r_center)
    y_axis.append(image.shape[0])
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
       # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        #print(level)
        lx.append(l_center)
        rx.append(r_center)
        y = int(image.shape[0]-level*window_height)
        y_axis.append(y)
    return window_centroids,y_axis,lx,rx 

#window_centroids,y_axis,lx,rx = find_window_centroids(top_down, window_width, window_height, margin)
#print(np.shape(window_centroids))
#print(window_centroids)
#print(y_axis)
# If we found any window centers
'''
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
    #If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results

    plt.figure()
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title('Orignal')
    ax1.imshow(warped)

    ax2.set_title('window fitting results')
    ax2.imshow(output)
'''




# In[12]:


# Fit a second order polynomial to pixel positions in each fake lane line
import numpy as np
import matplotlib.pyplot as plt

fit_left_buffer =[]
fit_right_buffer = []

def loco_position(pts,xm_per_pix,undistorted):
    # Locate the position of the car from the center
    position = undistorted.shape[1]/2
    left  = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    center = (left + right)/2
    #Convert into meters 
    result = (position - center)*xm_per_pix
    return result 

#ploty=[680, 600, 520, 440, 360, 280, 200, 120, 40]


def fitting_projection(undistorted,warped,ploty,leftx,rightx,rev_M):
    
    
    y_plotval = np.linspace(0, 100, num=101)*7.2
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*y_plotval**2 + left_fit[1]*y_plotval + left_fit[2] #calculate the curve 
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*y_plotval**2 + right_fit[1]*y_plotval + right_fit[2] #calculate the curve 
    
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 60
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    #print(np.shape(leftx))
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_fitx = left_fit[0]*y_plotval**2 + left_fit[1]*y_plotval + left_fit[2]
    right_fitx = right_fit[0]*y_plotval**2 + right_fit[1]*y_plotval + right_fit[2]
    
    tp1=480
    #tp2=600
    
    tp_1_leftx = left_fit[0]*tp1**2 + left_fit[1]*tp1 + left_fit[2]
    tp_1_rightx = right_fit[0]*tp1**2 + right_fit[1]*tp1 + right_fit[2]
    #print(tp_1_rightx-tp_1_leftx)
    
    #print(left_curverad, right_curverad)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    #n = np.array(ploty,dtype=np.float32)*ym_per_pix

    left_fit_cr = np.polyfit(np.array(lefty ,dtype=np.float32)*ym_per_pix, np.array(leftx,dtype=np.float32)*xm_per_pix, 2)
    right_fit_cr = np.polyfit(np.array(righty ,dtype=np.float32)*ym_per_pix, np.array(rightx,dtype=np.float32)*xm_per_pix, 2)
    # Calculate the new radii of curvature
    #print(left_fit_cr)
    #print(right_fit_cr)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_plotval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_plotval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    
    if(abs(np.mean(left_curverad) - np.mean(right_curverad))<400 and abs(tp_1_rightx-tp_1_leftx)>750 and abs(tp_1_rightx-tp_1_leftx)<900):
        if(len(fit_left_buffer)<4):
            fit_left_buffer.append(left_fitx)
            fit_right_buffer.append(right_fitx)
            #print(len(fit_left_buffer[0]))
        else:
            fit_left_buffer.pop(0)
            fit_right_buffer.pop(0)
            fit_left_buffer.append(left_fitx)
            fit_right_buffer.append(right_fitx)
    #print(np.mean(fit_left_buffer, axis =0))
    #print(right_fitx)
    left_fitx = np.mean(fit_left_buffer, axis =0)
    right_fitx = np.mean(fit_right_buffer, axis =0)
        
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#np.mean(fit_left_buffer, axis =0)np.mean(fit_right_buffer, axis =0)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx-10, y_plotval]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx+5, y_plotval])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, rev_M, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    
    pts = np.argwhere(newwarp[:,:,1])
    position = loco_position(pts,xm_per_pix,undistorted)
    
    # Put text on an image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Radius of Curvature: {} m".format(right_curverad)
    cv2.putText(result,text,(400,100), font, 1,(255,255,255),2)
    
    text = "Left position: {} m".format(-position)
    cv2.putText(result,text,(400,150), font, 1,(255,255,255),2)
 
    return result 
    #=====================================
#print(len(fit_left_buffer))
#print(revser_M_beffer)
#print(np.mean(revser_M_beffer, axis =0))
#def draw_area(undistorted, warped,left_fitx,right_fitx, y_plotval,rev_M):
    
    #plt.figure()
    #plt.imshow(result)
    


# In[13]:


image = mpimg.imread('test_images/test3.jpg') 
undistorted,mtx,dist = cal_undistort(image, objpoints, imgpoints)
result = pipeline(undistorted)
top_down, perspective_M,rev_M = corners_unwarp(result, mtx, dist)
window_centroids,y_axis,lx,rx = find_window_centroids(top_down, window_width, window_height, margin)
#print(np.mean(lx))
#print(np.mean(rx)-np.mean(lx))
result = fitting_projection(undistorted,top_down,y_axis,lx,rx,rev_M);  
plt.imshow(result)
#print(fit_left_buffer)
#print(np.shape(np.mean(fit_left_buffer, axis =0)))
#print(np.mean(fit_left_buffer, axis =0))
#print(np.shape(np.mean(fit_right_buffer, axis =0)))
#draw_area(undistorted,top_down,y_axis,left_fitx, right_fitx,rev_M);  

#np.mean(fit_left_buffer, axis =0),np.mean(fit_right_buffer, axis =0)

    
# Example values: 1926.74 1908.48


# In[14]:


# Define conversions in x and y from pixels space to meters

# Example values: 632.1 m    626.2 m


# In[ ]:





# In[15]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    undistorted,mtx,dist = cal_undistort(image, objpoints, imgpoints)
    result = pipeline(undistorted)
    top_down, perspective_M,revser_M = corners_unwarp(result,  mtx, dist)
    window_centroids,y_axis,lx,rx = find_window_centroids(top_down, window_width, window_height, margin)
    output = fitting_projection(undistorted,top_down,y_axis,lx,rx,revser_M); 
    #draw_area(undistorted,top_down,y_axis,lx,rx,revser_M); 
    return output


# In[ ]:


white_output = 'test_videos_output/output.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
#white_output = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1= VideoFileClip("./project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))





