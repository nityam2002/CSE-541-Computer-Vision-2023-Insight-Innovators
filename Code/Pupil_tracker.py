import cv2
import numpy as np
import matplotlib.pyplot as plt

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('WIN_20230413_19_22_37_Pro.mp4')

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# size = (frame_width, frame_height)

# result = cv2.VideoWriter('filename.avi',cv2.VideoWriter_fourcc(*'MJPG'),10, size)
def create_model(cx,cy):
    pupil_center = (int(cx),int(cy),0)  # Example pupil center
    vector_length = 100

    sphere_center = (450, 250,-150)
    # Calculate the direction of the gaze
    gaze_direction = np.array(pupil_center) - np.array(sphere_center)
    gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
    
    # Draw the gaze direction as a line starting from the sphere center
    line_start = sphere_center
    print(line_start)
    line_end = line_start + vector_length * gaze_direction
    # line_end.astype(int)
    # for i in range(len(line_end)):
    #     print('ok')
    #     line_end[i]=int(line_end[i])
    # print(line_end)
    tempp=[]
    (np.rint(line_end)).astype(int)
    for i in range(len(line_end)):
        tempp.append(int(line_end[i]))
    # print(tempp)
    # cv2.line(image, (line_start), tuple(tempp), (255, 0, 0), 2)
    x = [int(cx),sphere_center[0] ]
    y = [int(cy), sphere_center[1]]
    z = [0, -150]

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear() # Clears the previous image
    # Plot the line
    ax.plot(x, y, z)
    center = (sphere_center[0], sphere_center[1], -150)
    radius = 50
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + (radius * np.sin(u) * np.cos(v))
    y = center[1] + (radius * np.sin(u) * np.sin(v))
    z = center[2] + (radius * np.cos(u))

    # Plot the sphere
    ax.plot_surface(x, y, z, alpha=0.2)



    ax.set_xlim([0, 640])
    ax.set_ylim([0,480 ])
    ax.set_zlim([-150, 150])
    # ax.set_box_aspect([1, 1, 1])
    # Show the plot
    ax.view_init(elev=-90, azim=-90)



    
    # ax.imshow() # Loads the new image
    # plt.pause(.1) # Delay in seconds
    # fig.canvas.draw()


    fig = plt.gcf()
    fig.canvas.draw()
    plot_np = np.array(fig.canvas.renderer.buffer_rgba())
    cv2.imshow("Plot", plot_np)



def fit_rotated_ellipse_ransac(data,iter=50,sample_num=10,offset=80.0):

    count_max = 0
    effective_sample = None

    for i in range(iter):
        sample = np.random.choice(len(data), sample_num, replace=False)

        xs = data[sample][:,0].reshape(-1,1)
        ys = data[sample][:,1].reshape(-1,1)

        J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
        Y = np.mat(-1*xs**2)
        P= (J.T * J).I * J.T * Y

        # fitter a*x**2 + b*x*y + c*y**2 + d*x + e*y + f = 0
        a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

        # threshold 
        ran_sample = np.array([[x,y] for (x,y) in data if np.abs(ellipse_model(x,y)) < offset ])

        if(len(ran_sample) > count_max):
            count_max = len(ran_sample) 
            effective_sample = ran_sample

    return fit_rotated_ellipse(effective_sample)


def fit_rotated_ellipse(data):

    xs = data[:,0].reshape(-1,1) 
    ys = data[:,1].reshape(-1,1)

    J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=np.float))) )
    Y = np.mat(-1*xs**2)
    P= (J.T * J).I * J.T * Y

    a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
    theta = 0.5* np.arctan(b/(a-c))  
    
    cx = (2*c*d - b*e)/(b**2-4*a*c)
    cy = (2*a*e - b*d)/(b**2-4*a*c)

    cu = a*cx**2 + b*cx*cy + c*cy**2 -f
    w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
    h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

    ellipse_model = lambda x,y : a*x**2 + b*x*y + c*y**2 + d*x + e*y + f

    error_sum = np.sum([ellipse_model(x,y) for x,y in data])
    print('fitting error = %.3f' % (error_sum))

    return (cx,cy,w,h,theta)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

def getcoordinates(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image_gray,(3,3),0)
    ret,thresh1 = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
    
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#   cv2.imshow("fs",image)
    image = 255 - closing

    contours , hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('f',image)
    hull = []

    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False)) 
                    
#   cnt = sorted(hull, key=cv2.contourArea)
#   maxcnt = cnt[-1]
    for con in hull:
        approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con,True),True)
        area = cv2.contourArea(con)
        if(len(approx) > 10 and area > 1000):
            flag=1
            cx,cy,w,h,theta = fit_rotated_ellipse_ransac(con.reshape(-1,2))
            # xcoordinates.append(cx)
            # ycoordinates.append(cy)
            return cx,cy,w,h,theta,flag
    return 0,0,0,0,0,0


def get_3d_eye_center(video_stream):
    # Define the intrinsic parameters of the camera
    camera_matrix = np.array([
    [5.884907549799992239e+02, 0.000000000000000000e+00, 2.752269078707765857e+02],
    [0.000000000000000000e+00, 5.982396776710994573e+02, 3.318492284433417581e+02],
    [0.000000000000000000e+00 ,0.000000000000000000e+00 ,1.000000000000000000e+00]
    ]) 
    
    # Define the initial guess for the 3D center of the eye
    eye_center_3d = np.array([0, 0, -50])
    dist_= np.array([-2.178911976291353914e-01 ,5.431022054650431752e-01, 1.653612419798152627e-02 ,-1.469481625747546920e-02 ,-8.703212030088525175e-01])
    # Define the confidence factor for each frame
    alpha = 0.8
    
    # Loop over the frames in the video stream
    while True:
        # Capture a frame from the video stream
        ret, frame = video_stream.read()
        if not ret:
            break
        
        # Convert the frame to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame
        ellipse=[[0,0],[0,0],0]
        # Fit an ellipse to the pupil
        ellipse[0][0],ellipse[0][1],ellipse[1][0],ellipse[1][1],ellipse[2],_ = getcoordinates(gray)
        
        # Convert the ellipse parameters to the form required by cv2.projectPoints
        center, axes, angle = ellipse[0], (ellipse[1][0]/2, ellipse[1][1]/2), ellipse[2]
        center = tuple(map(int, center))
        axes = tuple(map(int, axes))
        
        # Define the extrinsic parameters of the camera based on the estimated pose of the head
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        
        # Project the 3D eye center onto the 2D image plane
        print('sddddddddd')
        eye_center_3d = eye_center_3d.astype('float64')
        camera_matrix = camera_matrix.astype('float64')
        eye_center_2d, _ = cv2.projectPoints(eye_center_3d, rvec, tvec, camera_matrix, dist_)
        print('=========++++++++++++++++++')
        eye_center_2d = tuple(map(int, eye_center_2d[0][0]))
        print('2deyey', eye_center_2d)
        # Calculate the vector from the pupil center to the eye center
        pupil_center = center
        pupil_vector = np.array([pupil_center[0]-eye_center_2d[0], pupil_center[1]-eye_center_2d[1], 0])
        print('pup vect', pupil_vector)
        # Update the 3D eye center estimate using the confidence-weighted formula
        eye_center_3d += alpha * (np.linalg.norm(pupil_vector) / camera_matrix[0][0]) * np.dot(np.linalg.inv(camera_matrix), pupil_vector)
        
    return eye_center_3d


# print('#D POINTSSSS',get_3d_eye_center(cap))
# Read until video is completed
xcoordinates= []
ycoordinates= []
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
      ret, frame = cap.read()
      cx,cy,w,h,theta, flag= getcoordinates(frame)
      if flag==1:
            cv2.ellipse(frame,(int(cx),int(cy)),(int(w),int(h)),theta*180.0/np.pi,0.0,360.0,(0,0,255),1)
            cv2.drawMarker(frame, (int(cx),int(cy)),(0, 0, 255),cv2.MARKER_CROSS,2,1)
            

            create_model(cx,cy)
      cv2.drawMarker(frame,(450, 250),(0, 255, 0),cv2.MARKER_CROSS,2,10)
      cv2.imshow('Output',frame)
      
              
              
            #   result.write(frame)
    # Press Q on keyboard to  exit
      
      if cv2.waitKey(25) & 0xFF == ord('q'):
           break   
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
# result.release()
# Closes all the frames
cv2.destroyAllWindows()