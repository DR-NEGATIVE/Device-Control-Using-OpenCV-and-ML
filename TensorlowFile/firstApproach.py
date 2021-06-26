import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import PIL
import cv2
from sklearn.metrics import classification_report
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.engine import training
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.keras.engine.input_layer import Input
training_data=[]
DataDirectory="D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/dataset/original_sign"
Classes=["0","1","2","3","4","5"]
INIT_LR = 1e-4
EPOCHS = 50 #20
BS = 16 # 32
img_size=224
for category in Classes:
    path=os.path.join(DataDirectory,category)
    class_label=int(category)
    for img in os.listdir(path):
        img_array= cv2.imread(os.path.join(path,img))
        #plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)) ## alternative  cv2.COLOR_BGR2GRAY
        #plt.show()
        #img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
        new_array=cv2.resize(img_array,(img_size,img_size))
        training_data.append([new_array,class_label])
import random
random.shuffle(training_data)
X=[]
y=[]
for features ,label in training_data:
    X.append(features)
    y.append(label)
X=np.array(X,dtype="float32")
Y=np.array(y)
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.20,stratify=Y,random_state=42)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
#base modal
baseModel = MobileNetV2(weights = "imagenet",include_top=False,input_tensor=Input(shape=(img_size,img_size,3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("hand.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
plt.show()