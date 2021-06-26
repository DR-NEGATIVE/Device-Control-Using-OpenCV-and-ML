#from typing_extensions import final
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
DataDirectory="D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/dataset/original_sign/"
Classes=["0","1","2","3","4","5"]
# INIT_LR = 1e-4
# EPOCHS = 50 #20
# BS = 16 # 32
# img_size=224
# for category in Classes:
#     path=os.path.join(DataDirectory,category)
#     class_label=int(category)
#     for img in os.listdir(path):
#         img_array= cv2.imread(os.path.join(path,img))
#         #plt.imshow(cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)) ## alternative  cv2.COLOR_BGR2GRAY
#         #plt.show()
#         #img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
#         new_array=cv2.resize(img_array,(img_size,img_size))
#         training_data.append([new_array,class_label])
# import random
# random.shuffle(training_data)
# X=[]
# y=[]
# for features ,label in training_data:
#     X.append(features)
#     y.append(label)
# X=np.array(X).reshape(-1,img_size,img_size,3)
# X=X/255.0 # normalization
# Y=np.array(y)
# transfer learning
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = training_datagen.flow_from_directory(
	DataDirectory,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#history= model.fit()
history = model.fit(train_generator, epochs=25, steps_per_epoch=655//126, validation_data = train_generator, verbose = 1, validation_steps=3)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()