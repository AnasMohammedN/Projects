import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("traffic_sign_model.keras")

# Class names
traffic_signs = {
0:"Speed limit (20km/h)",
1:"Speed limit (30km/h)",
2:"Speed limit (50km/h)",
3:"Speed limit (60km/h)",
4:"Speed limit (70km/h)",
5:"Speed limit (80km/h)",
6:"End of speed limit (80km/h)",
7:"Speed limit (100km/h)",
8:"Speed limit (120km/h)",
9:"No passing",
10:"No passing for vehicles over 3.5 tons",
11:"Right-of-way at intersection",
12:"Priority road",
13:"Yield",
14:"Stop",
15:"No vehicles",
16:"Vehicles over 3.5 tons prohibited",
17:"No entry",
18:"General caution",
19:"Dangerous curve left",
20:"Dangerous curve right",
21:"Double curve",
22:"Bumpy road",
23:"Slippery road",
24:"Road narrows on the right",
25:"Road work",
26:"Traffic signals",
27:"Pedestrians",
28:"Children crossing",
29:"Bicycles crossing",
30:"Beware of ice/snow",
31:"Wild animals crossing",
32:"End of all speed and passing limits",
33:"Turn right ahead",
34:"Turn left ahead",
35:"Ahead only",
36:"Go straight or right",
37:"Go straight or left",
38:"Keep right",
39:"Keep left",
40:"Roundabout mandatory",
41:"End of no passing",
42:"End of no passing by vehicles over 3.5 tons"
}

st.title("Traffic Sign Classification")

uploaded = st.file_uploader(
    "Upload a Traffic Sign Image",
    type=["jpg","png","jpeg"]
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")

    st.image(image, caption="Uploaded Image", width=250)

    img = image.resize((30,30))

    img = np.array(img)/255.0

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    class_id = np.argmax(prediction)

    confidence = np.max(prediction)*100

    st.success(f"Prediction : {traffic_signs[class_id]}")

    st.write(f"Confidence : {confidence:.2f}%")