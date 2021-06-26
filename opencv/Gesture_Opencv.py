import webbrowser
import cv2
import pyautogui as pag
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
np.set_printoptions(suppress=True)
# pre loading Model 
model = tensorflow.keras.models.load_model('D:\Device-Control-Using-OpenCV-and-ML\Device-Control-Using-OpenCV-and-ML\opencv\model\keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
pred=[]
def image_prediction(imageOpenCv):
    # Disable scientific notation for clarity
    #np.set_printoptions(suppress=True)
    #nonlocal model
    # Load the model
    #model = tensorflow.keras.models.load_model('D:\Device-Control-Using-OpenCV-and-ML\Device-Control-Using-OpenCV-and-ML\opencv\model\keras_model.h5')

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    # data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('D:\Device-Control-Using-OpenCV-and-ML\Device-Control-Using-OpenCV-and-ML\obc.jpg ')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # display the resized image
    #image.show()
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print("output")
    print(prediction)
    dito = {0: "5", 1: "2", 2: "3", 3: "4", 4: "1", 5: "fist"}
    l = -1
    indexfind = max(prediction[0])
    for i in  prediction[0]:
        l=l+1
        if indexfind == i:
            break
    pred.append(dito[l])
    print("output is :=> ", dito[l])
c=10
f=0
img=cv2.VideoCapture(0)
hand_roi=-1
hand_xml=cv2.CascadeClassifier('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/full_hand_gesture.xml')
while True:
    rt,frame=img.read()
    cv2.imwrite("obc.jpg",frame)
    image_prediction("obc.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    palm = hand_xml.detectMultiScale(gray, 1.1, 4)
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGBA2GRAY) #cv2.COLOR_RGB2BGR working well
    for (x,y,w,h) in palm:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+w,x:x+w]
        hand_palm= hand_xml.detectMultiScale(roi_gray)
        if len(hand_palm)==0:
            #c=3
            print("No hand detected")
        else:
            for (ex,ey,ew,eh) in hand_palm:
                hand_roi=roi_color[ey:ey+eh,ex:ex+ew]# gray,frame
                # #c=3
                # #if c==5:
                #     #pag.hotkey('win','1') # pyautogui test
                #  #   print("chrome command executed")
                #    # chromedir= 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
                #     #webbrowser.get(chromedir).open("https://github.com/DR-NEGATIVE")
                # if c%3==0:
                #     #c=10
                #     c=1
                #     print("Second command execution")
                #     #pag.hotkey('win','down','down','down','down')
                #     #pag.press('right')  # for ppt
                #     #pag.press('space')

    cv2.imshow("Gesture Detection ",frame)# gray,frame
    cv2.imshow("crop_part",hand_roi)
    if cv2.waitKey(25) & 0xFF == ord('q') or len(pred)>6:
        
        break

#img.realese()
cv2.destroyAllWindows()
one=0
two=0
five=0
for i in pred:
    if i==1:
        one+=1
    elif i==2:
        two+=1
    else:
        five+=1
if one>two and one>five:
    from opencv.Aircanvas import paintcv
    paintcv.Aircanva()
elif two>one and two>five:
    pass
else:
    from opencv.BrowserOption import controller
    controller.browser_control()