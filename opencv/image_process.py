import cv2
import numpy as np
import matplotlib.pyplot as plt
file="D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/left_hand.jpg"
hand_xml=cv2.CascadeClassifier('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/hand_detection.xml')

frame=cv2.imread(file)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
palm = hand_xml.detectMultiScale(gray, 1.1, 4)
plt.imshow(palm)
print(palm)
for (x,y,w,h) in palm:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=frame[y:y+w,x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    hand_palm= hand_xml.detectMultiScale(roi_gray)
    if len(hand_palm)==0:
        print("No hand detected")
    else:
        for (ex,ey,ew,eh) in hand_palm:
            hand_roi=roi_color[ey:ey+eh,ex:ex+ew]
    #plt.imshow(frame,(x,y),(x+w,y+w),)
#frame=cv2.resize(frame,(224,224))
print((len(frame[0]),len(frame)))
plt.imshow(frame) # detection with square part
plt.imshow(hand_roi) #crop detected part
plt.show()
ditto=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

frame=cv2.cvtColor(frame,cv2.COLOR_RGB2HLS)


cv2.imshow("frame",frame)
cv2.imshow("ditto",ditto)
cv2.waitKey(0)
cv2.destroyAllWindows()