
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os

class Custom_Network():
    @staticmethod
    def Custom_Network(input_shape,classes=100,weights="trained_model/vggnet.hdf5"):
        model = Sequential()
        model.add(Convolution2D(32,5,5,border_mode="valid",subsample=(2,2),input_shape=input_shape)) #output=((227-5)/2 + 1 = 112
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((112-2)/2 + 1 = 56

        model.add(Convolution2D(32,5,5,border_mode="same")) #output = 56
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64,3,3,border_mode="same"))  #output = 56
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((56-2)/2 + 1 = 28


        model.add(Convolution2D(64,3,3,border_mode="same"))  #output = 28
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64,3,3,border_mode="same")) #output= 28
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((28-2)/2 + 1 = 14

        model.add(Convolution2D(192,3,3,border_mode="same"))  #output =6
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(192,3,3,border_mode="valid"))  #output = ((6-3)/1) + 1 = 4 
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  #output=((4-2)/2 + 1 = 2 
        
        model.add(Flatten())
        
        model.add(Dense(output_dim=4096,input_dim=2*2*192))
        model.add(Activation('relu'))
        model.add(Dropout(0.4)) # for sec level
        
        model.add(Dense(output_dim=4096,input_dim=4096))
        model.add(Activation('relu'))
        #model.add(Dropout(0.4)) # for first level
        model.add(Dropout(0.4)) # for sec level
        
        model.add(Dense(output_dim=classes,input_dim=4096))
        model.add(Activation('softmax'))

        if os.path.isfile(weights):
            model.load_weights(weights)
            print("Model loaded")
        else:
            print("No model is found")

        return model
