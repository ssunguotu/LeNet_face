# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
# video
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np
from predict import norm_size
import imutils

 
class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":   #for tensorflow
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        #second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
    
    def detect_image(self, image, model):
        #load the image
        # image = cv2.imread(args["image"])
        orig = image.copy()
        
        # pre-process the image for classification
        image = cv2.resize(image, (norm_size, norm_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        # classify the input image
        result = model.predict(image)[0]
        #print (result.shape)
        proba = np.max(result)
        label = str(np.where(result==proba)[0])
        label = "{}: {:.2f}%".format(label, proba * 100)
        print(label)
        
        # if args['show']:   
            # draw the label on the image
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25),cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
        return output