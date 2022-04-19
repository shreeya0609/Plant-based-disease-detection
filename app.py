import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing import image
class_names =["Healthy","Infected"]
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # Plant disease detection
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)


def import_and_predict(image_data, model):
    size = (260, 260)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = image/255
    img = np.expand_dims(img, axis=0)
    # img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.

    #img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img)

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    if predictions >= 0.5:
        st.write('{:.2%} percent confirmed that is Infected'.format(predictions[0][0]))
    else:
        st.write('{:.2%} percent confirmed that this is Healthy'.format(1 - predictions[0][0]))