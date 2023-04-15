
# Project Overview: Pupil Tracking and Gaze Estimation

## Motivation

The ability to track the pupil and estimate gaze direction is essential for many applications, including human-computer interaction, virtual reality, and gaming. However, accurate and efficient pupil tracking and gaze estimation remain challenging tasks.

## Usage

Our project aims to develop a system for tracking the pupil and estimating gaze direction using two methods: UNet and RANSAC. We further built an eye model to estimate gaze direction based on the detected pupil position.

## Recent Papers

Some recent papers on pupil tracking and gaze estimation include:

- "Pupil Tracking for Eye Gaze Estimation Based on Convolutional Neural Networks" by Sugano et al. (2014)
- "Real-Time Eye Tracking and Blink Detection with OpenCV" by Gavrila and Schiele (2016)
- "Pupil Detection Algorithm Based on RANSAC Procedure" by Radu Gabriel et al.

## Our Approach

Our approach combines two methods for pupil tracking and gaze estimation: UNet and RANSAC. UNet is used to accurately segment the pupil in images, we created our own dataset for training the model. While RANSAC is used to estimate the center of the pupil and the direction of gaze based on the detected edge points. We also built an eye model to estimate gaze direction based on the detected pupil position. Our experiments show that our approach is accurate and efficient for pupil tracking and gaze estimation, with RANSAC being the more efficient method.
