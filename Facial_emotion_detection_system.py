#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('unzip -uq "/content/drive/MyDrive/Emotion_Detection_Dataset.zip" -d "/content/drive/MyDrive/Emotion_Detection_Dataset"')


# In[ ]:


# change directory to current working directory
import os
os.chdir("/content/drive/MyDrive/Emotion_Detection_Dataset/fer2013")


# # **Augment Training And Validation Dataset Images**

# In[ ]:


# import necessary libraries
import keras
from keras.preprocessing.image import ImageDataGenerator


# define model parameters
num_classes = 7
img_row,img_col = 48,48
batch_size = 32

# define training and validation dataset
training_data = "train"
validation_data = "validation"

# data augmentation on training set
data_aug_train = ImageDataGenerator(
    rescale= 1./255,
    rotation_range = 30,
    shear_range = 0.3,
    zoom_range = 0.3,
    width_shift_range = 0.4,
    height_shift_range = 0.4,
    horizontal_flip = True,
    fill_mode = "nearest")

# data augmentation on validation dataset
data_aug_validation = ImageDataGenerator(rescale = 1./255)

# identify training dataset classes from the training dataset folder
train_parm = data_aug_train.flow_from_directory(
    training_data, 
    color_mode = "grayscale",
    target_size = (img_row,img_col),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True)

# identify validation dataset classes from the validation dataset folder
validation_parm = data_aug_validation.flow_from_directory(
    validation_data, 
    color_mode = "grayscale",
    target_size = (img_row,img_col),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = True)

train_parm


# In[ ]:


print(train_parm)


# # **Buiding our CNN model**

# In[ ]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import ReLU
from keras.layers.core import Activation,Flatten,Dropout,Dense

model = Sequential()

# First Layer Group Conv => Relu => Conv =>Relu =>POOL =>Dropout
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (img_row,img_col,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding = 'same',input_shape = (img_row,img_col,1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# second Layer Group Conv => Relu => Conv =>Relu =>POOL =>Dropout
model.add(Conv2D(128,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# third Layer Group Conv => Relu => Conv =>Relu =>POOL =>Dropout
model.add(Conv2D(256,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding = 'same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# fourth layer group - Fully connected =>relu =>dropout
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# fifth layer group relu => Dropout
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# sixth layer group
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())


# In[ ]:


from keras.utils.vis_utils import plot_model
plot_model(model,to_file = 'faceial_expression_architecture.png')


# # **Training The Model**

# In[ ]:


from keras.optimizers import Adam

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])

training_sample = 28709
validation_sample = 3589
epochs =120

fitted_model = model.fit_generator(train_parm,steps_per_epoch=training_sample//batch_size,epochs=epochs,validation_data=validation_parm,
                                   validation_steps = validation_sample//batch_size)


# In[ ]:


model.save("facial_emotion_detection.h5")


# In[ ]:


# get the class label
validation_parm = data_aug_validation.flow_from_directory(
    validation_data, 
    color_mode = "grayscale",
    target_size = (img_row,img_col),
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle = False)
class_labels = validation_parm.class_indices
print(class_labels)
print(class_labels.items())
class_labels = {a: k for k,a in class_labels.items()}
classes = list(class_labels.values())
print(classes)
print(class_labels)


# In[5]:


from tensorflow.keras.models import load_model
import cv2
import numpy as np


# In[3]:


model = load_model("facial_emotion_detection.h5")
face_det_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")



vid_source = cv2.VideoCapture(0)
text_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
rect_color_dict = {0:(0,0,255),1:(2,156,173),2:(193,217,219),3:(0,255,0),4:(0,255,255),5:(79,102,100),6:(255,0,0)}

while(True):
    ret,img = vid_source.read()
    grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_det_classifier.detectMultiScale(grayscale_img,1.3,5)
    
    for (x,y,w,h) in faces:
        face_img = grayscale_img[y:y+h,x:x+w]
        resized_img = cv2.resize(face_img,(48,48))
        normalized_img = resized_img/255.0
        reshaped_img = np.reshape(normalized_img,(1,48,48,1))
        reshaped_img = reshaped_img.astype('float32')
        result = model.predict(reshaped_img)
        
        label = np.argmax(result,axix = 1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),rect_color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),rect_color_dict[label],-1)
        cv2.putText(img,text_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
        
    cv2.imshow('live feed',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
vid_source.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




