from dotenv import load_dotenv
load_dotenv() ##loading all the environent variables

import streamlit as st ##allows you to build interactive, web-based applications quickly and easily
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##function to load Gemini pro model and get responses
model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(prompt):
    response = model.generate_content(prompt)
    return response.text

##initializing the streamlit app
st.set_page_config(page_title="QnA Demo")
st.header("Gemini LLM Application")

input = st.text_input("Input: ",key="input")
submit = st.button("Ask the question!")

##when submit is clicked
if submit:
    response = get_gemini_response(input)
    st.subheader("The response is: ")
    st.write(response)