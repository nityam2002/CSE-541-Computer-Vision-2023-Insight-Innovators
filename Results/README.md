

In this project we have implemted a system for tracking the pupil and estimating gaze using two methods: UNet and RANSAC.

## Methods Used

- **UNet**: A deep learning method that uses a U-shaped convolutional neural network to segment the pupil in images.
- **RANSAC**: A robust statistical method used to estimate the center of the pupil and the direction of gaze based on the detected edge points.

## Results

Both methods were evaluated on a dataset of eye images, and their accuracy and performance were compared.

- UNet was found to be accurate in detecting the pupil but was slower than RANSAC. [IOU accuracy: 98.6%]
- RANSAC was also accurate in detecting the pupil and estimating the gaze direction and was more efficient than UNet. [MSE Erroe: 0.1]

Overall, both methods were able to track the pupil and estimate gaze accurately, but RANSAC was found to be more efficient.

