from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(image,prompt):
    if prompt != "" :
        response = model.generate_content([prompt,image])
    else :
        response = model.generate_content(image)
    return response.text

##initialize streamlit
st.set_page_config(page_title="Visionary")
st.header("Welcome!")

input = st.text_input("Input prompt: ",key="input")

##image upload
file = st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
if file is not None :
    img = Image.open(file)
    st.image(img,caption="Uploaded Image.",use_container_width=True)

submit = st.button("Ask")

if submit :
    response = get_gemini_response(img,input)
    st.subheader(">>>")
    st.write(response)