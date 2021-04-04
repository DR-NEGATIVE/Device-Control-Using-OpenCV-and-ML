import cv2
vid = cv2.VideoCapture(0)
hand_xml=cv2.CascadeClassifier('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/hand_detection.xml')

while True:
    rt,frame=vid.read()
    
    temp=frame
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours,hierachy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(temp, contours, -1, (0,255,0), 3)

    palm = hand_xml.detectMultiScale(im_bw, 1.1, 4)
    
    cv2.imshow('Test',im_bw)
    cv2.imshow("color",temp)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        
        break
vid.realese()
cv2.destroyAllWindows()