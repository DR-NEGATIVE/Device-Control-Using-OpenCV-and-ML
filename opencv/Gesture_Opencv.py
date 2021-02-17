import cv2
c=0
img=cv2.VideoCapture(0)
while True:
    rt,frame=img.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR) #cv2.COLOR_RGB2BGR working well
    cv2.imshow("ok",frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        
        break

img.realese()
cv2.destroyAllWindows()