import webbrowser
import cv2
import pyautogui as pag
c=10
f=0
img=cv2.VideoCapture(0)
hand_roi=-1
hand_xml=cv2.CascadeClassifier('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/hand_detection.xml')
while True:
    rt,frame=img.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    palm = hand_xml.detectMultiScale(gray, 1.1, 4)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY) #cv2.COLOR_RGB2BGR working well
    for (x,y,w,h) in palm:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+w,x:x+w]
        hand_palm= hand_xml.detectMultiScale(roi_gray)
        if len(hand_palm)==0:
            print("No hand detected")
        else:
            for (ex,ey,ew,eh) in hand_palm:
                hand_roi=roi_color[ey:ey+eh,ex:ex+ew]# gray,frame
                c=c-1
                if c==5:
                    #pag.hotkey('win','1') # pyautogui test
                    print("chrome command executed")
                    chromedir= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
                    webbrowser.get(chromedir).open("https://github.com/DR-NEGATIVE")
                if c==0:
                    print("close command executed")
                    pag.hotkey('win','down','down','down')

    cv2.imshow("ok",frame)# gray,frame
    cv2.imshow("crop_part",hand_roi)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        
        break

img.realese()
cv2.destroyAllWindows()