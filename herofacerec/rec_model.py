from PIL import Image
import os
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import MaxPool2D,Dense,Dropout,Conv2D,Flatten,BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
num_classes = 3 # number of heroes
x = [] #images of heroes
y = [] #categories of hero names

hero_names = {'Chiranjeevi':0,'Nagarjuna':1,'RamCharan':2}

def feed_images(folder,name):
    for image in os.listdir(folder):
        loaded_image = Image.open(os.path.join(folder,image))
        loaded_image_resize = Image.Image.resize(loaded_image,[100,100])
        a = np.array(loaded_image_resize)/255
        x.append(a)
        y.append(hero_names[name])

        image_flipped = cv2.flip(a,1)
        x.append(image_flipped)
        y.append(hero_names[name])
        image_blurred = cv2.blur(a,(2,2))
        x.append(image_blurred)
        y.append(hero_names[name])
        image_flipped_blurred = cv2.blur(image_flipped,(2,2))
        x.append(image_flipped_blurred)
        y.append(hero_names[name])

feed_images('C:/Users/raghu/Desktop/cnn/heroes/chiru','Chiranjeevi')
feed_images('c:/users/raghu/desktop/cnn/heroes/nag','Nagarjuna')
feed_images('c:/users/raghu/desktop/cnn/heroes/ram','RamCharan')

Y = np_utils.to_categorical(y,num_classes=num_classes)
X = np.array(x)
model = Sequential()
model.add(Conv2D(10,kernel_size=(1,1),strides=(1,1),input_shape=(100,100,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(10,activation="relu"))
model.add(Dense(10,activation="relu"))
model.add(Dense(num_classes,activation="softmax"))
model.summary()
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])
model_graph = model.fit(X,Y,epochs=10,batch_size=100,verbose=1,validation_split=0.05)
plt.plot(model_graph.history['accuracy'],label="train")
plt.plot(model_graph.history['val_accuracy'],label="test")
plt.show()
model.save('heropred.h5')
model_pred= load_model('heropred.h5')
load_image = Image.open('c:/users/raghu/desktop/cnn/heroes/guess/guess6.jpg')
load_image_resize = Image.Image.resize(load_image,[100,100])
load_image_normalize = (np.array(load_image_resize))/255
load_image_reshape = load_image_normalize.reshape(1,100,100,3)
prediction = model.predict_classes(load_image_reshape)
if prediction == 0:
    print("chiru")
elif prediction == 1:
    print("nag")
elif prediction == 2:
    print("ram")
