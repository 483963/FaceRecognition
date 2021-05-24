#!/usr/bin/env python
# coding: utf-8

# In[21]:


#importing modules for training the network
import tensorflow as tf #tensorflow import
from keras.models import Model #to construct the model 
from keras.preprocessing import image #for procesing the images
from keras.models import Sequential #for sequential model construction
from keras.layers import Input , Conv2D, Dense ,Flatten , Dropout , Activation, MaxPooling2D #various layers for CNN


# In[22]:


import scipy.io as sio #reading and writing images
import numpy as np #for linear algebra operations
import matplotlib.pyplot as plt #plotting graphs
import keras.optimizers as optimizers #optimizer to be used during training


# In[23]:


model = Sequential()  
    
# Convolution -> ReLU -> MaxPool2D -> Dropout
model.add(Conv2D(128, (3, 3),padding='valid', input_shape=(224, 224,3 ))) #128 (3x3)filters and input shape of 224 by 224
model.add(Activation('relu')) #adding activation of ReLU
model.add(MaxPooling2D(pool_size=(2, 2))) # 2 by 2 maxpooling to reduce the feaures
model.add(Dropout(0.3)) #add dropout to reduce overfitting model

# Convolution
model.add(Conv2D(64, (3, 3))) #64 (3x3) filters other parameters save as above convolution block
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Flatten
model.add(Flatten()) #flatten the input into single dimentional vector representation

# Fully connection
model.add(Dense(512))  # unit between[#input, #output]
model.add(Activation('relu'))
    
# initialize output layer
model.add(Dense(15)) #output layer with 15 neurons as we have 15 classes
model.add(Activation('softmax')) #as multiclass classification problem so SoftMax activation function
model.summary() #to print the model architecture


# In[24]:


# compile
lr=0.001 #learning rate
decay=1e-6 #decay
momentum=0.9 # momentum
sgd = optimizers.SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True) #using SGC optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy']) #compiling model with categorical cross entropy and SGD with momentum


# In[25]:


# Part 2 - Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator #generate image batches

#use the image data generator to import the images from the dataset
#data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[26]:


#setting training and testing directories
train_path = 'C://Users//user//Desktop//Infento//Project 3. Face Recognition//dataset_cnn//train//' 
test_path = 'C://Users//user//Desktop//Infento//Project 3. Face Recognition//dataset_cnn//test//'
#using flow_from_directory for creating the images from local directories
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=2,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=2,
                                            class_mode='categorical',
                                            shuffle=False)


# In[27]:


#fit the model
nb_train_samples=120 #num of training samples
nb_validation_samples=45 #num of testing samples
batch_size=2 #how many image to process at once
steps_per_epoch = len(training_set)//batch_size #num of epochs to train the model


#fir model and record the history in a variable
history = model.fit(training_set,
                              validation_data=test_set,
                              epochs=100,
                              steps_per_epoch=nb_train_samples // batch_size,
                              validation_steps=nb_validation_samples // batch_size)


# In[28]:


#PLOTTING accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'] , label = 'train acc')
plt.plot(history.history['val_accuracy'] , label = 'val acc')
plt.legend()
plt.show()

#PLOTTING loss
import matplotlib.pyplot as plt
plt.plot(history.history['loss'] , label = 'train loss')
plt.plot(history.history['val_loss'] , label = 'val loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




