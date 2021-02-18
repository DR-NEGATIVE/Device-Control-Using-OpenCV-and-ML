import cv2
c=0
img=cv2.VideoCapture(0)
hand_xml=cv2.CascadeClassifier('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/hand_detection.xml')
while True:
    rt,frame=img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    palm = hand_xml.detectMultiScale(gray, 1.1, 4)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY) #cv2.COLOR_RGB2BGR working well
    for (x,y,w,h) in palm:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # gray,frame
    cv2.imshow("ok",frame)# gray,frame
    if cv2.waitKey(25) & 0xFF == ord('q'):
        
        break

img.realese()
cv2.destroyAllWindows()