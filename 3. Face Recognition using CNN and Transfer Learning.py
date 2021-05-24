#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary modules
import tensorflow as tf
from keras.applications.vgg16 import VGG16 #VGG16 pretrained on ImageNet will be imported
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input, Lambda ,Dense ,Flatten , Dropout , GlobalAveragePooling2D


# In[2]:


#vgg 16 model
classifier_vgg16 = VGG16(input_shape= (224,224,3),include_top=False,weights='imagenet') #initiate model with 224x224x3 input size
#not train top layers
for layer in classifier_vgg16.layers:
    layer.trainable = False #freeze the top layers
classifier_vgg16.summary() #print summary


# In[3]:


# #not train top layers
# for layer in classifier_vgg16.layers:
#     layer.trainable = False


# In[3]:


#adding extra layers for our classes (15 in our case)
main_model = classifier_vgg16.output #get output from VGG-16
main_model = GlobalAveragePooling2D()(main_model) #Add global average pooling layer
main_model = Dense(512,activation='relu')(main_model) #add 512 neurons fully connected layer 
main_model = Dense(256,activation='relu')(main_model) #add 256 neurons fully connected layer 
main_model = Dropout(0.5)(main_model) #dropout to reduce overfitting
main_model = Dense(15,activation='softmax')(main_model) #final layer with 15 neurons to reconnize the subjects


# In[4]:


#compiling
model = Model(inputs = classifier_vgg16.input , outputs = main_model) #define model
model.summary() #print summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #compile with ADAM and Categorical Cross Entropy


# In[5]:


# Part 2 - Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator #generate batches of images for training and testing

#use the image data generator to import the images from the dataset
#data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)


# In[6]:


#setting the train and testing images paths
train_path = 'C://Users//user//Desktop//Infento//Project 3. Face Recognition//dataset_cnn//train//' 
test_path = 'C://Users//user//Desktop//Infento//Project 3. Face Recognition//dataset_cnn//test//'
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=2,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=2,
                                            class_mode='categorical',
                                            shuffle=False)


# In[10]:


#fit the model
#it will take some time to train
nb_train_samples=120
nb_validation_samples=45
batch_size=2
steps_per_epoch = len(training_set)//batch_size #num of epochs

#fit the model
history = model.fit(training_set,
                              validation_data=test_set,
                              epochs=50,
                              steps_per_epoch=nb_train_samples // batch_size,
                              validation_steps=nb_validation_samples // batch_size)


# In[ ]:





# In[11]:


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




