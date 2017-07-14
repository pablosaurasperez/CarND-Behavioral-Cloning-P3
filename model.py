import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
tf.python.control_flow_ops = tf


samples = []

#Open csv file and read its lines
with open('data_training/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)
        
from sklearn.model_selection import train_test_split
from random import shuffle

print(len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.3)

#Use a generator to load data samples and process them on the fly
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: #Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
        
            #Inputs
            images = []
            #Outputs
            angles = []
            correction = 0.2
            
            for batch_sample in batch_samples:
                #Load Center/Left/Right images
                for i in range(3):
                    #Take the center/left/right image path
                    source_path = batch_sample[i]
                    #For the Inputs (features) (Images)
                    #take only the file name
                    filename = source_path.split('/')[-1]
                    #Look for the file name in the image folder
                    current_path = 'data_training/IMG/' + filename
                    
                    #Once I have the image path, I can use opencv to load it
                    image = cv2.imread(current_path)
                    #Once I loaded the image I can append it to my list of images
                    images.append(image)
                    
                    #For the outputs (Measurements) (Steering angles)
                    #I can do something similar for the steering angles (only interested in this parameter now)
                    #Center Steering
                    if i == 0:
                        angle = float(batch_sample[3])
                    #Left Steering
                    elif i == 1:
                        angle = float(batch_sample[3]) + correction
                    #Right Steering
                    else:
                        angle = float(batch_sample[3]) - correction
            
                    angles.append(angle)
                
            #Flip images so that I have also right turns
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                #And now I flip
                image_flipped = np.fliplr(image)
                angle_flipped = -angle
                augmented_images.append(image_flipped)
                augmented_angles.append(angle_flipped)
                
            #Now I can convert Features and Outputs in numpy arrays (this is the formast Keras requires)
            #Because I augmented the set, I have 3 times the original samples
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
                    

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

#Test NVIDIA Network

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Prepocess data. 
#Cropping
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160, 320, 3)))
#Normalize and mean center
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

#NVIDIA Net

#5 convolutional layers
#Stride 2x2
model.add(Convolution2D(24, 5, 5 , subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5 , subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5 , subsample=(2,2), activation='relu'))
#No stride
model.add(Convolution2D(64, 3, 3 , activation='relu'))
model.add(Convolution2D(64, 3, 3 , activation='relu'))

model.add(Flatten())
model.add(Dropout(0.5))
#3 fully connected layers
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))


#Output layer
model.add(Dense(1))
#end LeNet

#MSE (predicted-GT) because is regression network instead of classification (Cross_Entropy)
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.switch_backend('agg')
fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig.savefig('Loss.png')

#Save the trained model so that I can download it to my local machine an see if it's able to go autonomous
model.save('model.h5')
print("Model Saved")
exit()

