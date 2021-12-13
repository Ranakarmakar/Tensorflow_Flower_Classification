import streamlit as st
import classify
import time
from PIL import Image
import numpy as np
import webbrowser

st.set_page_config(layout="wide", page_title="Flower Classification", page_icon="Flower")

st.title("Flower Classification Model [Tensorflow]")
st.markdown("Rana Karmakar [Website](https://rana-reflective-porcupine-pf.eu-gb.mybluemix.net)")
c1, c2 = st.columns(2)
with c1:
    uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False)
    class_names = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        if st.button('   Submit  '):
            results = classify.predict(image)
            with st.spinner('Loading Result...'):
                time.sleep(2)
                st.markdown("This Image most likely belongs to")
                st.subheader(
                    " {} with a {:.2f} percent confidence."
                        .format(class_names[np.argmax(results)], 100 * np.max(results)))
                st.write(results)
                with open("model_summary.png", "rb") as file:
                    btn = st.download_button(
                        label="Model Summary",
                        data=file,
                        file_name="model_summary.png",
                        mime="image/png"
                    )
    else:
        with open("training_results.png", "rb") as file:
            btn = st.download_button(
                label="Model Performance",
                data=file,
                file_name="training_results.png",
                mime="image/png"
            )

if uploaded_file is None:
    st.header("“Love is the flower you’ve got to let grow.” - John Lennon")
with c2:
    st.write("Demonstration on How tensorflow works with Image Data, You can Download Sample Images [from here]("
             "https://unsplash.com/s/photos/daisy%2C-dandelion%2C-roses%2C-sunflowers%2C"
             "-tulips)")
    if uploaded_file is not None:
        st.image(uploaded_file)
    else:
        st.subheader("Please Upload Images of Daisy, Dandelion, Roses, Sunflowers, Tulips Flowers only")


# Conclusion
ex1, ex2, ex3 = st.columns(3)
with ex1:
    with st.expander("About The Project"):
        st.markdown("Flower Classification")
        st.markdown(
            "This Project shows how to classify images of flowers. It creates an image classifier using a "
            "tf.keras.Sequential model, and loads data using tf.keras.utils.image_dataset_from_directory.")
        st.markdown("See Other Awesome Apps")
        url = 'https://share.streamlit.io/ranakarmakar/streamlit_movie_recommendation/main/app.py'
        url1 = 'https://share.streamlit.io/ranakarmakar/brain_tumor_classification/main/tumor.py'
    if st.button("Movie Recommendation System"):
        webbrowser.open_new_tab(url)

with ex3:
    with st.expander("About Developer"):
        st.markdown("Rana Karmakar")
        st.write("I have a deep interest in Artificial Intelligence and Machine Learning ever since I got to know "
                 "about it; the sci-fi films, comics and stories have always fascinated me. I love to learn new "
                 "skills to keep myself up-to-date with the corporate world. I believe in maintaining a work-life "
                 "balance while learning and working upon different fields of interest.I have had a taste of many "
                 "different technologies: creating Websites, Software Development, Data Analysis, "
                 "creating Machine Learning models, Cloud Computing and Developing complex programs and more. "
                 "Feel free to give Feedback at ranakarmakar027@gmail.com")
    if st.button("Brain Tumor Classification"):
        webbrowser.open_new_tab(url1)
with ex2:
    with st.expander("Contact"):
        st.markdown("ranakarmakar027@gmail.com")
        st.write("[Website](https://rana-reflective-porcupine-pf.eu-gb.mybluemix.net/)")
        st.write("[LinkedIn](https://www.linkedin.com/in/rana-karmakar-0972641a6)")
        st.write("Other Apps[Movie Recommendation System]("
                 "https://share.streamlit.io/ranakarmakar/streamlit_movie_recommendation/main/app.py)")
        st.write("[Brain Tumor Detection]("
                 "https://share.streamlit.io/ranakarmakar/brain_tumor_classification/main/tumor.py)")
