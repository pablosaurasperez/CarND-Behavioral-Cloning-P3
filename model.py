import csv
import cv2
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

lines = []

#Open csv file and read its lines
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
#Inputs
images = []
#Outputs
measurements = []
correction = 0.2

for line in lines:
    #Load Center/Left/Right images
    for i in range(3):
        #Take the center/left/right image path
        source_path = line[i]
    
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
        #Center Steering
        if i == 0:
            measurement = float(line[3])
        #Left Steering
        elif i == 1:
            measurement = float(line[3]) + correction
        #Right Steering
        else:
            measurement = float(line[3]) - correction
            
        measurements.append(measurement)
    
#Flip images so that I have also right turns
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    #And now I flip
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    augmented_images.append(image_flipped)
    augmented_measurements.append(measurement_flipped)
    
#Now I can convert Features and Outputs in numpy arrays (this is the formast Keras requires)
#Because I augmented the set, I have double the original samples
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


#Test NVIDIA Network

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Prepocess data. 
#Cropping
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
#Normalize and mean center
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#NVIDIA

#5 convolutional layers
#Stride 2x2
model.add(Convolution2D(24, 5, 5 , subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5 , subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5 , subsample=(2,2), activation='relu'))
#No stride
model.add(Convolution2D(64, 3, 3 , activation='relu'))
model.add(Convolution2D(64, 3, 3 , activation='relu'))

model.add(Flatten())

#3 fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))

#Output layer
model.add(Dense(1))
#end LeNet

#MSE (predicted-GT) because is regression network instead of classification (Cross_Entropy)
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=3)

#Save the trained model so that I can download it to my local machine an see if it's able to go autonomous
model.save('model.h5')
exit()

