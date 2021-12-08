# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 23:34:27 2021

@author: LEGION
"""


# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""


# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""

# IMPORT MODULES
import sys
from os.path import join
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

#from tensorflow.python.keras.applications.resnet50 import preprocess_input
#from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
#from tensorflow.python.keras.applications import ResNet50

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
#from keras.utils import np_utils, to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#from keras.applications import ResNet50

#% check if gpu available or not 
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#% check if gpu available or not 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Brain-Tumor-Classification-DataSet-master\Training'
valid_path = 'Brain-Tumor-Classification-DataSet-master\Testing'

# add preprocessing layer to the front of VGG





# Convoluted Base MODEL

conv_base = ResNet50(weights='imagenet',
include_top=False,
input_shape=(224, 224, 3))

print(conv_base.summary())


# Make the conv_base NOT trainable:

for layer in conv_base.layers[:]:
   layer.trainable = False

print('conv_base is now NOT trainable')

from glob import glob
folders = glob('Brain-Tumor-Classification-DataSet-master/Training/*')

# MODEL

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(len(folders), activation='softmax'))

print(model.summary())



# for i, layer in enumerate(conv_base.layers):
#    print(i, layer.name, layer.trainable)
   
   
   
# Compile frozen conv_base + my top layer
# tell the model what cost and optimization method to use

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

print("model compiled")
print(model.summary())


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Brain-Tumor-Classification-DataSet-master\Training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Brain-Tumor-Classification-DataSet-master\Testing',
                                            
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')



print('------- Evalute Model -------------------')

from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix

Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)


print('Confusion Matrix')
cm=confusion_matrix(test_set.classes, y_pred)
print(cm)

print('-------- plot confusion_matrix ------')
import seaborn as sns
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title(' Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2','3'])
ax.yaxis.set_ticklabels(['0','1','2','3'])
## Display the visualization of the Confusion Matrix.
plt.show()



print('----------  Classification Report -------------')
import os 
target_names = os.listdir(valid_path)
#target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(test_set.classes,y_pred,target_names=target_names))


# save Model
import tensorflow as tf
from keras.models import load_model
model.save('facefeatures_new_model.h5')

#% check if gpu available or not 
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

#% check if gpu available or not 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'Brain-Tumor-Classification-DataSet-master\Training'
valid_path = 'Brain-Tumor-Classification-DataSet-master\Testing'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False
  

  
  # useful for getting number of classes
folders = glob('Brain-Tumor-Classification-DataSet-master/Training/*')
  

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Brain-Tumor-Classification-DataSet-master\Training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Brain-Tumor-Classification-DataSet-master\Testing',
                                            
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

'''r=model.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000)'''

# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')



print('------- Evalute Model -------------------')

from sklearn.metrics import classification_report, confusion_matrix,plot_confusion_matrix

Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)


print('Confusion Matrix')
cm=confusion_matrix(test_set.classes, y_pred)
print(cm)

print('-------- plot confusion_matrix ------')
import seaborn as sns
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title(' Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2','3'])
ax.yaxis.set_ticklabels(['0','1','2','3'])
## Display the visualization of the Confusion Matrix.
plt.show()



print('----------  Classification Report -------------')
import os 
target_names = os.listdir(valid_path)
#target_names = ['class 0', 'class 1', 'class 2','class 3']
print(classification_report(test_set.classes,y_pred,target_names=target_names))


# save Model
import tensorflow as tf
from keras.models import load_model
model.save('facefeatures_new_model.h5')