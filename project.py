import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import random

# Calibrate camera
def calibrate(img, objpoints, imgpoints):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Find corners of the chessboard
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist

# Return the undistorted image
def calc_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Unwarp an image
def unwarp(img, nx, ny, objpoints, imgpoints, offset):
    # Undistort the image
    undist = calc_undistort(img, objpoints, imgpoints)
    
    # Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    
    # Search for corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        # Draw corners
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        img_size = (gray.shape[1], gray.shape[0])
        
        # Get the outer four corners detected for source points
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        
        # Use offset to choose four corners for destination points
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                         [img_size[0]-offset, img_size[1]-offset],
                         [offset, img_size[1]-offset]])
        
        # Calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Warp the image
        warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    else:
        warped, M = None
    return warped, M

# Function to plot and save images
def plot_imgs(imgs, titles, figsize=(24, 9), cmap='gray', save=False):
    num_imgs = len(imgs)
    f, axes = plt.subplots(1, num_imgs, figsize=figsize)
    f.tight_layout()
    for i in range(num_imgs):
        axes[i].imshow(imgs[i], cmap=cmap)
        axes[i].set_title(titles[i], fontsize=50)
    if save == True:
        plt.savefig('./output_images/' + titles[-1] + '.png', bbox_inches='tight')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Gradient threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) if orient == 'x' else cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    absolute = np.absolute(sobel)
    scaled = np.uint8(255*absolute/np.max(absolute))
    grad_binary = np.zeros_like(scaled)
    grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return grad_binary

# Magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    mag_binary = np.zeros_like(scaled)
    mag_binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
    return mag_binary

# Direction of the gradient
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    grad = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad)
    dir_binary[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    return dir_binary

# Function to threshold the S channel
def S_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

# Final pipeline for thresholding images
def threshold_pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    # Convert to HSV color space and exclude the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# Define source and destination points
src = np.float32([[600, 448], [686, 448], [1120, 720], [202, 720]])
dst = np.float32([[350, 0], [900, 0], [900, 720], [350, 720]])
 
# Warp an image    
def warp(img, reverse=False):
    img_size = (img.shape[1], img.shape[0])
    
    # Compute perspective transform and inverse
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Allow for transforming from warped image, back to original
    if reverse == True:
        warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
        return warped
    else:
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return warped

def find_lane_lines(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    
    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    
    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Color lane lines red and blue
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return out_img, ploty, left_fitx, right_fitx, left_fit, right_fit

# Search for lane lines within a margin around the lines from the previous frame
def find_more_lane_lines(img, left_fit, right_fit):
    # Find line pixels in the next frame
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Return None if not enough pixels were found within the margin
    if lefty.shape[0] < 10 or righty.shape[0] < 10:
        return None, None, None, None, None, None, None
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return ploty, left_fitx, right_fitx, left_fit, right_fit

# Alternative to sliding window search
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(img, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(img.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(img.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(img[int(img.shape[0]-(level+1)*window_height):int(img.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,img.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,img.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def draw_lines(img):
    # window settings
    window_width = 50 
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channle 
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((img,img,img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((img,img,img)),np.uint8)

    return output

# Measure radius of curve in pixels
def measure_curve_pixels(ploty, left_fitx, right_fitx):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_fit = np.polyfit(ploty, left_fitx, 2)
    right_fit = np.polyfit(ploty, right_fitx, 2)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return(left_curverad, right_curverad)

# Measure radius of curve in meters
def measure_curve(ploty, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 27/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return (left_curverad, right_curverad)

def distance_from_center(image_width, pts):
    # Get position of center camera, should be center of image
    position = image_width/2
    left_base = np.min(pts[(pts[:,1] < position) & (pts[:,0] > 700)][:,1])
    right_base = np.max(pts[(pts[:,1] > position) & (pts[:,0] > 700)][:,1])
    
    # Expected center of lane, half distance between left and right corners
    center = (left_base + right_base)/2
    
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7/550  
    return (position - center) * xm_per_pix

# Fill in the lane with color and convert back to the original perspective
def draw_on_road(undist, warped, ploty, left_fitx, right_fitx, radius_left, radius_right):
    # Create an image to draw lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_(pts), (0, 255, 0))
    
    # Warp the blank back into original image space using inverse perspective matrix
    new_warp = warp(color_warp, src, dst, reverse=True)
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 0.7, new_warp, 0.3, 0)
    
    # Calculate distance from center in meters
    imwidth = undist.shape[1]
    pts = np.argwhere(new_warp[:, :, 1])
    distance = distance_from_center(imwidth, pts)
    
    # Write curve on image
    result = cv2.putText(result, 
                "Left Radius: {0} m, Right Radius: {1} m".format(int(radius_left), int(radius_right)), 
                (5,40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    
    # Write distance from center on image
    result = cv2.putText(result, 
                "Distance from center: {0} m".format(str(distance)), 
                (5,70), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    return result

# First attempt at pipeline
def pipeline1(img):
    # Define source and destination points
    src = np.float32([[600, 448], [686, 448], [1120, 720], [202, 720]])
    dst = np.float32([[350, 0], [900, 0], [900, 720], [350, 720]])

    # Warp images
    warped = warp(img, reverse=False)

    # Create binary warped images
    binary_warped = threshold_pipeline(warped, 
        s_thresh=(170, 255), 
        sx_thresh=(20, 100))

    # Find lines
    out_img, ploty, left_fitx, right_fitx = find_lane_lines(binary_warped)

    # Measure the curve of the road
    curve = measure_curve(ploty, left_fitx, right_fitx)

    # Fill in lane green and undistort
    new_img = draw_on_road(img, binary_warped, ploty, left_fitx, right_fitx, radius_left, radius_right)

    return new_img, curve

# Line class for keeping track of recent measurements
class Line():
    def __init__(self):
        # Was the line detected in the last iteration?
        self.detected = False

        # Number of iterations to smooth over
        self.n = 3

        # Most recent coefficients
        self.current = [0, 0, 0]

        # Difference in coefficients between last and new fit
        self.diffs = [0, 0, 0]

        # Polynomial coefficients with length n
        self.A = []
        self.B = []
        self.C = []

        # Average coefficients from last n iterations
        self.avg_A = 0
        self.avg_B = 0
        self.avg_C = 0

        # Radius of curvature
        self.rad_of_curve = None

    def add_avg_n_coefficients(self, A, B, C, radius):
        # Most recent coefficients
        self.current = [A, B, C]

        # Difference in coefficients between last n and new
        self.diffs = [A - self.avg_A, B - self.avg_B, C - self.avg_C]

        # Add new coeffiecients to array
        self.A.append(A)
        self.B.append(B)
        self.C.append(C)

        # Remove oldest coefficient if number of stored coefficients is greater than n
        if len(self.A) > self.n:
            self.A.pop(0)
            self.B.pop(0)
            self.C.pop(0)

        # Update average of coefficients
        self.avg_A = np.mean(self.A)
        self.avg_B = np.mean(self.B)
        self.avg_C = np.mean(self.C)

        # Radius of curvature
        self.rad_of_curve = radius

        return(self.avg_A, self.avg_B, self.avg_C)

    def get_coefficients(self):
        return(self.avg_A, self.avg_B, self.avg_C, self.diffs, self.rad_of_curve)

left_line = Line()
right_line = Line()

# Final pipeline    
def pipeline(img):
    # Undistort images
    undist = calc_undistort(img, objpoints, imgpoints)
    
    # Define source and destination points
    src = np.float32([[600, 448], [686, 448], [1120, 720], [202, 720]])
    dst = np.float32([[350, 0], [900, 0], [900, 720], [350, 720]])

    # Warp images
    warped = warp(undist, src, dst)

    # Create binary warped images
    binary_warped = threshold_pipeline(warped)
    
    # If a line was not detected:
    if left_line.detected == False:
        # Find lines blindly
        out_img, ploty, left_fitx, right_fitx, left_fit, right_fit = find_lane_lines(binary_warped)
        
        # Measure the radius of the curve of the road
        radius = measure_curve(ploty, left_fitx, right_fitx)
        radius_left = radius[0]
        radius_right = radius[1]
        
        # Add coefficients and radii to lines
        left_coeff = left_line.add_avg_n_coefficients(left_fit[0], left_fit[1], left_fit[2], radius_left)
        right_coeff = right_line.add_avg_n_coefficients(right_fit[0], right_fit[1], right_fit[2], radius_right)
        
        left_line.detected = True
    
    else:
        # Get last coefficients
        a_left, b_left, c_left, diffs_left, radius_left = left_line.get_coefficients()
        a_right, b_right, c_right, diffs_right, radius_right = right_line.get_coefficients()
        left_coeff = [a_left, b_left, c_left]
        right_coeff = [a_right, b_right, c_right]
        
        # Try to find lines within margin of last detected line
        margin = 100
        out_image, window_img, ploty, left_fitx, right_fitx, left_fit, right_fit = find_more_lane_lines(binary_warped, left_coeff, right_coeff)
        
        # Calculate radius of each line
        radius = measure_curve(ploty, left_fitx, right_fitx)
        radius_left = radius[0]
        radius_right = radius[1]
        
        # If a line was detected within the margin and radii are reasonably close to parallel:
        if (left_fit != None) and ((radius_right < radius_left + 50) and (radius_right > radius_left - 50)):
            # Add coefficients and radii to lines
            left_coeff = left_line.add_avg_n_coefficients(left_fit[0], left_fit[1], left_fit[2], radius_left)
            right_coeff = right_line.add_avg_n_coefficients(right_fit[0], right_fit[1], right_fit[2], radius_right)
            left_fitx = left_coeff[0]*ploty**2 + left_coeff[1]*ploty + left_coeff[2]
            right_fitx = right_coeff[0]*ploty**2 + right_coeff[1]*ploty + right_coeff[2]
        else:
            # Use average coefficients from last n frames to plot lines
            left_fitx = left_line.avg_A*ploty**2 + left_line.avg_B*ploty + left_line.avg_C
            right_fitx = right_line.avg_A*ploty**2 + right_line.avg_B*ploty + right_line.avg_C
            left_line.detected = False
            
    # Fill in lane green and undistort
    new_img = draw_on_road(img, binary_warped, ploty, left_fitx, right_fitx, radius_left, radius_right).astype(np.uint8)
    
    # Write curve on image
    new_img = cv2.putText(new_img, 
                "Left: {0} Right: {1}".format(str(radius_left), str(radius_right)), 
                (5,5), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    return new_img