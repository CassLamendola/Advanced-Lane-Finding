# Advanced Lane Finding Project
---
### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distortion.png "Undistorted"
[image2]: ./output_images/warped.png "Warped Chessboard"
[image3]: ./output_images/Undistorted.png "Undistorted Road"
[image4]: ./output_images/Warped.png "Warped Road Example"
[image5]: ./output_images/Thresholds.png "Binary Examples"
[image6]: ./output_images/Thresholded_S.png "Binary Color"
[image7]: ./output_images/combined_thresholds.png "Combined Thresholds"
[image8]: ./output_images/Thresholded_and_Warped.png "Thresholded and Warped"
[image9]: ./output_images/find_lines.png "Sliding Window"
[image10]: ./output_images/Output.png "Output Example"
[image11]: ./output_images/final_test_imgs "Final Test Images"
[video1]: ./project_result.mp4 "First Attempt"
[video2]: ./final_project_result.mp4 "Final Video"
[video3]: ./challenge_video_result.mp4 "Challenge Video"
[video4]: ./harder_challenge_result.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients. I created a function `calibrate` (lines 8-17 in project.py) which converts an image to grayscale, finds chessboard corners using `cv2.findChessboardCorners()`, and then uses the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image1]

I also unwarped a chessboard image using the `cv2.warpPerspective()` function. This function requires source `src` and destination `dst` points. I calculated the source points using the outermost corners that were found using `cv2.findChessboardCorners`, and the destination points were obtained from the size of the image and an offset of 100. Here is the result:
![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
I used the same object points and image points that were found during camera callibration to correct the distortion in road images. I created a function,  `calc_undistort()` (lines 20-22 in project.py) which takes as inputs an image, `mtx`, and `dist` (found during camera calibration) and outputs an undistorted image. The function uses `cv2.undistort()`


####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (lines 68-137 in project.py). The function that is used in my final pipeline is `threshold_pipeline()`. It convers an image to HLS colorspace and combines Sobel x and the S channel for thresholding. Here's an example of my output for this step.
![alt text][image7]
Here are some examples of different techniques:
![alt text][image5]
![alt text][image6]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()` (lines 143-157 in project.py) The `warp()` function takes as inputs an image , as well as source (`src`) and destination (`dst`) points and an optional arguement `reverse` which is by default set to `False`.  I chose the hardcode the source and destination points based off of a test image of a straight road

I found four points in the road which would form a rectangle from a bird's eye view.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 448      | 350, 0        | 
| 668, 448      | 900, 0        |
| 1120, 720     | 900, 720      |
| 202, 720      | 350, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify lane lines, I combined thresholding and perspective transform. Here are a few examples:
![alt text][image8]

Then I created a function `find_lane_lines()` (lines 159-248 in project.py) that takes a warped and thresholded image as input and outputs an image with distinct colored lane lines, the formulas for the lines, and the coefficients. This function uses a histogram to blindly search the image for lines by finding the two peaks of greatest white pixels within an expected range for lane lines. It starts at the bottom of the image at these two peaks and then a sliding 'window' moves up the image recentering on white pixels foudn within the window. In each window, the white pixels that were found are added to an array of points. Then a line is fit to the array of points and those points are colored in red for the left line and blue for the right line. Here is an example of its output.
![alt text][image9]

Another function was created to search within a margin of the last detected line. This function is called `find_more_lane_lines()` (lines 250-283 in project.py). This function takes as inputs a a warped and thresholded image and coefficients for left and right lane lines. It then searches for points within a margin of those lines. If not enought points are found, it outputs None, otherwise it outputs the formulas and coefficients for both lines.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculate the radius of the curve with a function called `measure_curve()` (lines 370-386 in project.py) which takes as input the coefficients from the left and right lines and outputs the curve radius for each. In order to measure the curve in meters instead of pixels, I needed to calculate the number of pixels per meter in the example images. A standard lane is 3.7 meters, which is 700 pixels in my warped image. The warpee image also has 3 dashed lines. Each line is about 3 meters long with about 6 meters between lines. Therefore the lenght of the road is about 27 meters which equals 720 pixels.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step with the function `draw_on_road()` (lines 388-411 in project.py). This function takes in an undistorted image, a warped image, and the formulas for the left and right lines. It then fills in the space between the lines in the warped image and unwarps the image with the `warp()` function. I set `reverse=True` in `warp()` so that it would use the inverse Matrix. Here is an example of my result on a test image:

![alt text][image10]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

My first attempth at the pipeline only uses the sliding window technique for finding the lines in each frame. This pipeline does very well on the test images. That makes sense, since the images are in no particular order, therefore we don't expect consecutive frames to have a similar curvature. But the video is not perfect. It had trouble where the lines are difficult to see. For these cases, I needed to do a targeted search for the lane based on the location of the lines in the last few frames. That way, if the lines are lost/hard to find, I could use the average from the last few frames and try again in the next frame. 

To accomplish this, I created a Line class that stores information about the lines from the last few frames and updated my pipeline to make use of the `find_more_lane_lines()` function.
Here's [link to my final video result](./final_project_result.mp4.zip)
You can also check out the videos in my [python notebook](https://github.com/CassLamendola/Advanced-Lane-Finding/blob/master/project.ipynb)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When my pipeline is tested on the challenge videos, it fails in several places. It is unable to detect the lane in variable light conditions or areas of the road with dark and light. Also, the output is a little wobbly. I could fix this with smoothing over more frames and implementing more checks in my pipeline. For example, I could check that the lines found from the blind search (`find_lane_lines()`) are reasonable based off of the last several frames. I could also experiment with even more thresholding combinations to find a result that works well under more difficult conditions.

