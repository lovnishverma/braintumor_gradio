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
    # If it's a NumPy array, use it directly
    if isinstance(img, np.ndarray):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        # Convert Gradio image data to bytes
        img_bytes = img.read()

        # Convert to NumPy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Crop and preprocess the grayscale image
    img_processed = preprocess_imgs([img_gray], (224, 224))

    return img_processed

# Handle binary decision
def binary_decision(confidence):
    return 1 if confidence >= 0.5 else 0

def predict_braintumor(img):
    # Preprocess the image
    img_processed = preprocess_image(img)

    # Make prediction
    pred = braintumor_model.predict(img_processed)

    # Handle binary decision
    confidence = pred[0][0]
    return "Brain Tumor Not Found!" if binary_decision(confidence) == 1 else "Brain Tumor Found!"

def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
                      extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
        set_new.append(new_img)
    return np.array(set_new)

# Gradio interface
iface = gr.Interface(
    fn=predict_braintumor,
    inputs="image",
    outputs="text",
    live=True  # Allows real-time updates without restarting the app
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
        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Make prediction
        pred = braintumor_model.predict(img_array)

        # Handle binary decision
        confidence = pred[0][0]
        result = "Brain Tumor Not Found!" if binary_decision(confidence) == 1 else "Brain Tumor Found!"

        # Display the prediction result
        st.write(result)
# Gradio
title = 'Face Recognition with Emotion and Sentiment Detector'

description = gr.Markdown(
                """Ever wondered what a person might be feeling looking at their picture? 
                 Well, now you can! Try this fun app. Just upload a facial image in JPG or
                 PNG format. Voila! you can now see what they might have felt when the picture
                 was taken.
                 This is an updated version of Facial Expression Classifier: 
                 https://huggingface.co/spaces/schibsted/facial_expression_classifier
                 """).value

article = gr.Markdown(
             """**DISCLAIMER:** This model does not reveal the actual emotional state of a person. Use and 
             interpret results at your own risk! It was built as a demo for AI course. Samples images
             were downloaded from VG & AftenPosten news webpages. Copyrights belong to respective
             brands. All rights reserved.
             
             **PREMISE:** The idea is to determine an overall sentiment of a news site on a daily basis
             based on the pictures. We are restricting pictures to only include close-up facial
             images.
             
             **DATA:** FER2013 dataset consists of 48x48 pixel grayscale images of faces. There are 28,709 
             images in the training set and 3,589 images in the test set. However, for this demo all 
             pictures were combined into a single dataset and 80:20 split was used for training. Images
             are assigned one of the 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
             In addition to these 7 classes, images were re-classified into 3 sentiment categories based
             on emotions:
             
             Positive (Happy, Surprise)
             
             Negative (Angry, Disgust, Fear, Sad)
             
             Neutral (Neutral)
             
             FER2013 (preliminary version) dataset can be downloaded at: 
             https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
             
             **EMOTION / SENTIMENT MODEL:** VGG19 was used as the base model and trained on FER2013 dataset. Model was trained
             using PyTorch and FastAI. Two models were trained, one for detecting emotion and the other
             for detecting sentiment. Although, this could have been done with just one model, here two
             models were trained for the demo.
             
             **FACE DETECTOR:** Darknet with YOLOv3 architecture was used for face detection. Reach out to me for full details.
             In short, any image is first sent through darknet. If face is detected, then it is passed through emotion/sentiment 
             model for each face in the picture. If a person is detected rather than a face, the image is cropped and run through
             face detector again. If a face is detected, then it is passed through emotion/sentiment model. In case face is not
             detected in an image, then the entire image is evaluated to generate some score. This is done because, I couldn't
             figure out how to pipe None/blank output to Gradio.Interface(). There maybe option through Gradio.Blocks() but was
             too lazy to go through that at this stage. In addition, the output is restricted to only 3 faces in a picture. 
             """).value

enable_queue=True
examples=[["examples/1_no.jpeg"], ["examples/2_no.jpeg"], ["examples/3_no.jpg"], ["examples/Y1.jpg"], ["examples/Y2.jpg"], ["examples/Y3.jpg"]],

gr.Interface(fn = predict, 
             inputs = gr.Image(), 
             outputs = [gr.Image(shape=(48, 48), label='Person 1'), 
                        gr.Label(label='Emotion - Person 1'), 
                        gr.Label(label='Sentiment - Person 1'),
                        gr.Image(shape=(48, 48), label='Person 2'), 
                        gr.Label(label='Emotion - Person 2'), 
                        gr.Label(label='Sentiment - Person 2'),
                        gr.Image(shape=(48, 48), label='Person 3'), 
                        gr.Label(label='Emotion - Person 3'), 
                        gr.Label(label='Sentiment - Person 3'),], #gr.Label(),
             title = title,
             examples = examples,
             description = description,
             article=article,
             allow_flagging='never').launch(enable_queue=enable_queue)