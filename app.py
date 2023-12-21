import streamlit as st

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
     """
)

# Sidebar with information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a trained model to predict brain tumors from images. "
    "The model is based on VGG16 architecture."
)

# Streamlit components below the Gradio interface
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Image.", use_column_width=True)

    # Perform prediction when the "Predict" button is clicked
    if st.button("Predict"):
        # Perform your prediction logic here
        # ...

# Embed Gradio interface within Streamlit using an HTML iframe
gradio_link = "http://localhost:7860/"
st.markdown(f"### Gradio Interface")
st.markdown(f"*(Note: Gradio Interface may take a moment to load.)*")
st.markdown(f'<iframe src="https://huggingface.co/spaces/LovnishVerma/medical" width="100%" height="600px" frameborder="0"></iframe>', unsafe_allow_html=True)
