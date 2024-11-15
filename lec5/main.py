from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from PIL import Image
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) ##configuring genai

model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(input,image,prompt):
    response = model.generate_content([input,image[0],prompt]) ##input is used to curate the ai as per the need
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        byte_data = uploaded_file.getvalue() ##read the file into the bytes, because APIs dont directly accept the images
        image_parts = [
            {
            "mime_type" : uploaded_file.type,## Gets the mime type of the uploaded file
            "data" : byte_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
        

##initialising streamlit
st.set_page_config(page_title="Invoice Reader") ##setting the page configs, here only title is set
st.header("GEMINI INVOICE READER") ##This the Header which will be shown in on the page
input = st.text_input("Input",key="input") ##This will create a text input box on the page
uploaded_file = st.file_uploader("Choose the image type of the Invoice...",type=["jpeg","jpg","png"]) ##will create a file uploader on the page of the streamit

image = "" ##Empty image variable to later

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption="Uploaded File",use_column_width=True) ##creates a section to display the image on the page streamlit page
    
submit = st.button("Send")

input_prompt = """
you are an expert in undertsanding invoices. We will provide you a invoice as an image and
you will have to answer any questions based on the uploaded invoice image
"""

if submit:
    image_data = input_image_details(uploaded_file)
    response = get_gemini_response(input_prompt,image_data,input)
    st.subheader("Response: ")
    st.write(response) ##writes the response on the streamlit page