
# -*- coding: utf-8 -*-
"""
@author: Krish.Naik
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf

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
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
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