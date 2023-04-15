
### Extract_Video_To_Frames.py
#### extracts individual frames from a video file. This script is useful for projects that require annotations or analysis of individual frames within a video.

### annotation_tool.py
#### Script for the annotaion tool that was used to annotate the images which were extracted from the video. It takes five or more points from the user to annotate a pupil and uses fitting algorithm to fit an ellipse to those points. It saves two files for each image. One that contains all the points on the ellipse and second that has 5 parameters of the ellipse from which we can reconstruct the shape.

### Unet_Model.py
#### It implements the U-Net architecture using TensorFlow layers. The U-Net architecture is a widely used convolutional neural network (CNN) for image segmentation tasks. The script is designed to train the U-Net model on a dataset of images and their corresponding masks. The images and masks should have the same size of 480x640 pixels. The U-Net model is trained for a binary semantic segmentation.

### Unet_Model_Test.py
#### For testing the trained U-net model via Transfer Learning. Uses pre-trained weights for performing the binary semantic segmentation.

### Pupil_tracker.py
#### It uses RANSAC (Random Sample Consensus) to detect the pupil in an eye image. This script is useful for projects that require accurate and robust pupil detection in eye-tracking applications. The script estimates the eye center based on the detected pupil and uses this information to create an eye model that visualizes the gaze direction of the person in the image.




