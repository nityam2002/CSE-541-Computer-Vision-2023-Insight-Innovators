import cv2
import os

# Open the video file
cap = cv2.VideoCapture('/home/devyashshah/Downloads/CLG_COURSES/SEM VI/CV4/divyashree.mp4')

# Create directory to store frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Set counter for file naming
count = 1

while True:
    # count = 0
    # Read the next frame
    ret,frame = cap.read()
    # print(success)\
    if ret == True:
 
        image = frame
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # If no more frames, break out of loop
    if not ret:
        break
    if count % 2 == 0:
        # Save the frame to file. We have taken every alternate frame.
        cv2.imwrite('frames/G%d.jpg' % (count//2), gray)

    count += 1

cap.release()
        
