#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('model.h5')

#Name of Classes
CLASS_NAMES = ['Cat','Dog']

#Setting Title of App
st.title("Cat vs Dog prediction")
st.markdown("Upload an image")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...")
submit = st.button('Predict')
#On predict button click
if submit:


    if dog_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)
        #Make Prediction
        Y_pred = model.predict(x=opencv_image)
        if Y_pred[0][0] > Y_pred[0][1]:
            st.title(str("It is a cat"))
        else:
            st.title(str("It is a dog"))

        st.title(str(Y_pred[0]))
