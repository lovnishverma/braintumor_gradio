import streamlit as st
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# Loading Models
braintumor_model = load_model('models/brain_tumor_binary.h5')

# Configure Streamlit
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":brain:",
    layout="wide",
)

# Title and description
st.title("Brain Tumor Detection App")
st.markdown(
    """Curious about detecting brain tumors in medical images? 
     Give this app a try! Upload an MRI image in JPG or
     PNG format, and discover whether it shows signs of a brain tumor.
     This is an updated version of the Brain Tumor Classifier: 
     [Kaggle Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/)
     """
)

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a trained model to predict brain tumors from images. "
    "The model is based on VGG16 architecture."
)

# Function to preprocess the image
def preprocess_image(img):
    if isinstance(img, np.ndarray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_bytes = img.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img_processed = preprocess_imgs([img_gray], (224, 224))
    return img_processed

# Handle binary decision
def binary_decision(confidence):
    return 1 if confidence >= 0.5 else 0

def predict_braintumor(img):
    img_processed = preprocess_image(img)
    pred = braintumor_model.predict(img_processed)
    confidence = pred[0][0]
    return "Brain Tumor Not Found!" if binary_decision(confidence) == 1 else "Brain Tumor Found!"

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Gradio interface
iface = gr.Interface(
    fn=predict_braintumor,
    inputs="image",
    outputs="text",
    examples=[
        ["examples/1_no.jpeg"],
        ["examples/2_no.jpeg"],
        ["examples/3_no.jpg"],
        ["examples/Y1.jpg"],
        ["examples/Y2.jpg"],
        ["examples/Y3.jpg"],
    ],
    live=True
)

# Display Gradio interface
iface.launch()

# Streamlit components below the Gradio interface
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image.", use_column_width=True)

    # Perform prediction when the "Predict" button is clicked
    if st.button("Predict"):
        img_array = preprocess_image(uploaded_file)
        pred = braintumor_model.predict(img_array)
        confidence = pred[0][0]
        result = "Brain Tumor Not Found!" if binary_decision(confidence) == 1 else "Brain Tumor Found!"

        # Display the prediction result with confidence
        st.success(result)
        st.write(f"Confidence: {confidence:.2%}")
