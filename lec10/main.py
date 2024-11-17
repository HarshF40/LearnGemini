from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_prompt,image):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_prompt,image])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        ##read the file into bytes
        img_data = uploaded_file.getvalue()
        img_part = {
                "mime_type" : uploaded_file.type,
                "data" : img_data
            }
        return img_part
    else :
        FileNotFoundError("File Not Uploaded!")

##streamlit app

st.set_page_config(page_title="Calories Advisor")
st.header("Geminutritionist")
uploaded_file = st.file_uploader("Upload an Image: ",type=["jpg","jpeg","png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file) ##image will store pixel data, this will be used to access and the edit the image
    st.image(image,caption="Uploaded Image",use_container_width=True)

submit = st.button("Count the Calories")

input_prompt="""
You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----
"""

if submit :
    image_data = input_image_setup(uploaded_file)
    response = get_gemini_response(input_prompt,image_data)
    st.header("The Response: ")
    st.write(response)