import csv
import cv2
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

lines = []

#Open csv file and read its lines
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Inputs
images = []
#Outputs
measurements = []
for line in lines[1:]:
    #Take the center image path
    source_path = line[0]
    
    #For the Inputs (features) (Images)
    #take only the file name
    filename = source_path.split('/')[-1]
    #Look for the file anme in the image folder
    current_path = '../data/IMG/' + filename
   
    #Once I have the image path, I can use opencv to load it
    image = cv2.imread(current_path)
    #Once I loaded the image I can append it to my list of images
    images.append(image)
    
    #For the outputs (Measurements) (Steering angles)
    #I can do something similar for the steering angles (only interested in this parameter now)
    measurement = float(line[3])
    measurements.append(measurement)
    
#Now I can convert Features and Outputs in numpy arrays (this is the formast Keras requires)
X_train = np.array(images)
y_train = np.array(measurements)


#Test LeNet

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Prepocess data. Normalize and mean center
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

#LeNet
model.add(Convolution2D(6, 5, 5 , activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Convolution2D(16, 5, 5 , activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(120))
model.add(Activation('relu'))

model.add(Dense(84))
model.add(Activation('relu'))

model.add(Dense(1))
#end LeNet

#MSE (predicted-GT) because is regression network instead of classification (Cross_Entropy)
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

#Save the trained model so that I can download it to my local machine an see if it's able to go autonomous
model.save('model.h5')

