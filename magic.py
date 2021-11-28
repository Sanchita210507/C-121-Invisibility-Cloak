import cv2
import time
import numpy as np

# video codec compress video data and encode into a format that can later be decoded and played back.
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# the third paramter is fps that is frame rate of the video string
outputFile = cv2.VideoWriter('Output.avi',fourcc,20.0,(640,418))

#Starting the webcam
cap = cv2.VideoCapture(0)

#Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
bg = 0

#Capturing background for 60 frames
for i in range(60):
    ret,bg = cap.read()

bg = np.flip(bg,axis=1)

# reading the captured frame until the camera is open.
while(cap.isOpened()):
    ret,img = cap.read()
    print(img)
    if not ret:
        break
    img = np.flip(img,axis = 1)
    # converting the colour from bgr to hsv(hue, saturation, value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowerRed = np.array([0,120,50])
    upperRed = np.array([10,255,255])

    mask1 = cv2.inRange(hsv, lowerRed, upperRed)

    lowerRed = np.array([170,120,70])
    upperRed = np.array([180,255,255])

    mask2 = cv2.inRange(hsv, lowerRed, upperRed)
    mask1 = mask1 + mask2

    # morphologyEx method is of the class image processing which is used to perform operations on a given image.
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))

    # selecting only the part that does not have mask 1 and saving in mask 2(filtering out red colour from the image).
    mask2 = cv2.bitwise_not(mask1)

    #Keeping only the part of the images without the red color
    res1 = cv2.bitwise_and(img,img,mask = mask2)
    #Keeping only the part of the images with the red color
    res2 = cv2.bitwise_and(bg, bg, mask = mask1)

    #function helps in transition of img to another. In order to blend this image, we can add weights & define the transparency and translucency of the images.
    finalOutput = cv2.addWeighted(res1 , 1 , res2 , 1 , 0)
    outputFile.write(finalOutput)
    cv2.imshow('My magic',finalOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
outputFile.release()
cv2.destroyAllWindows()