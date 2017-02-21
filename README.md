## Advanced Lane Finding

In this project, the goal was to write a software pipeline to identify the lane boundaries in a video. Check out my [writeup](https://github.com/CassLamendola/Advanced-Lane-Finding/blob/master/writeup.md) for this project for more details.  

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`. The images in `test_images` were used for testing my pipeline on single frames.

I've saved examples of the output from each stage of my pipeline in the folder called `ouput_images`, and included a description in the writeup for the project of what each image shows. The video called `project_video.mp4` is the video my pipeline was designed to work on.  

The `challenge_video.mp4` video was provided as an extra (and optional) challenge to test my pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!
