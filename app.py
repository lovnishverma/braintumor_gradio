import streamlit as st
import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Loading Models
braintumor_model = load_model('models/brain_tumor_binary.h5')

# Configure Streamlit
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon=":brain:",
    layout="wide",
)

# Title and description
st.title('Brain Tumor Detection App')
st.markdown(
    """Curious about detecting brain tumors in medical images? 
     Give this app a try! Upload an MRI image in JPG or
     PNG format, and discover whether it shows signs of a brain tumor.
     This is an updated version of the Brain Tumor Classifier: 
     [Kaggle Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/)
     """, unsafe_allow_html=True  # Make sure to allow HTML rendering
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
    return binary_decision(confidence)

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

# Streamlit components
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg"])

# Display Gradio interface
st.markdown("<h1 style='text-align: center;'>Gradio Interface</h1>", unsafe_allow_html=True)

# Create a placeholder for the Gradio interface
gradio_placeholder = st.empty()

# Create a custom HTML block for title and description
html_block_title_description = gr.HTML(
    f"<h2>Brain Tumor Detection App</h2>"
    "<p>Curious about detecting brain tumors in medical images? "
    "Give this app a try! Upload an MRI image in JPG or "
    "PNG format, and discover whether it shows signs of a brain tumor. "
    "This is an updated version of the Brain Tumor Classifier: "
    "<a href='https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/' target='_blank'>Kaggle Dataset</a></p>"
)

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

# Display title and description in Gradio interface
iface.launch(gradio_placeholder, html_block_title_description)

# Streamlit components below the Gradio interface
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image.", use_column_width=True)

    # Perform prediction when the "Predict" button is clicked
    if st.button("Predict"):
        result = predict_braintumor(uploaded_file)

        # Display the prediction result with confidence
        if result == 1:
            st.success("Brain Tumor Found!")
        else:
            st.success("Brain Tumor Not Found!")
