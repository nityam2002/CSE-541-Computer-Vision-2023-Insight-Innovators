from tensorflow import keras
from keras.utils import normalize
import cv2
import numpy as np
import random
import os
from matplotlib import pyplot as plt
# img = cv2.imread("Original\F139.jpg")
# print(img.shape)

# img = np.expand_dims(img, axis=0)
# print(img.shape)
# model = simple_unet_model(img.shape[0], img.shape[1], img.shape[2])
# model.load_weights('mitochondria_test.hdf5')
model = keras.models.load_model('unet_pupil', compile=False)







cap = cv2.VideoCapture("aastha.mp4")
# cap = cv2.VideoCapture(0)
pupil_center_x, pupil_center_y = 0,0

while True:
    # cv2.waitKey(10)

    ret,frame = cap.read()
    if ret == True:
        # cv2.imshow('efs',frame)
        # mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # # Define the center and the axes of the ellipse
        # center = (frame.shape[1] // 2 +100, frame.shape[0]// 2 +35)  # (x, y) format
        # axes = (frame.shape[1] // 4, frame.shape[0] // 4)  # (major_axis, minor_axis) format

        # # Draw the ellipse on the mask
        # cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        # # Apply the mask to the image
        # result = cv2.bitwise_and(frame, frame, mask=mask)

        # # Set the non-elliptical part to white
        # result[~mask.astype(bool)] = (255, 255, 255)


        image = frame


        blurredA1=cv2.blur(image,(3,3))
    


        
        image = np.expand_dims(blurredA1, axis=0)
        # image = np.expand_dims(image, axis=3)
        print(image.shape)
        pred = model.predict(image)
        print(pred.shape)
        pred= np.squeeze(pred, axis=0)
        pred= np.squeeze(pred, axis=2)
        print(pred.shape)
        cv2.imshow("sdc",pred)
        # _,contours = cv2.findContours(np.array(pred, dtype=np.uint8),  mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        # contours, _ = cv2.findContours(pred=np.array(pred, dtype=np.int32), method = cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            print(i)
            if len(i)>4:
                x,y,w,h = cv2.boundingRect(i)
                # res_img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                frame = cv2.circle(frame, (int(x+h//2), int(y+w//2)),3 , (255, 0, 0), 2)
        cv2.imshow('ssvg',frame)
        cv2.waitKey(1)