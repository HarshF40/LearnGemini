##Resume ATS
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from PIL import Image
import pdf2image
import google.generativeai as genai
import io
import base64

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_text,pdf_content,prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text,pdf_content[0],prompt]) ##pdf_content is a list which has an image of the resume
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## convert the pdf to image
        images = pdf2image.convert_from_bytes(uploaded_file.read()) ##reads the pdf files binary data and then convert that data into images using pdf2image.convert_from_bytes
        first_page = images[0]

        ##convert to bytes
        img_byte_arr = io.BytesIO() ##in-memory stream that behaves like a file, its a temporary file, its an temporary storage for the image in binary format so that it can be later converted into base64, so img_byte_arr is a temporary file to store the binary data
        first_page.save(img_byte_arr,format="jpeg") ##Saves the extracted first-page image to the img_byte_arr in JPEG format, the img_byte_arr now contains the binary represnatation of the image
        img_byte_arr = img_byte_arr.getvalue() ##retrives the binary image data from the in-memory stream
    
        ##Encodes the binary image data into a Base64 string for further use, such as embedding the image in HTML or sending it via an API

        ##base64 enc
        pdf_parts = [
            {
                "mime_type" : "image/jpeg",
                "data" : base64.b64encode(img_byte_arr).decode() #encode to base64
                }
            ]
        return pdf_parts
    else :
        FileNotFoundError("File Not Found")

##streamlit
st.set_page_config(page_title="Resume ATS")
st.header("ATS")
input_text = st.text_area("Job Description",key="input")
uploaded_file = st.file_uploader("Uploade Resume(pdf)...",type=["pdf"])

##submit buttons
submit1 = st.button("Tell me about the Resume")
submit2 = st.button("Percentage match")

if uploaded_file is not None:
    st.write("File Uploaded Successfully!")

    input_prompt1 = """
     You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
      Please share your professional evaluation on whether the candidate's profile aligns with the role. 
     Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """

input_prompt2 = """
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
    the job description. First the output should come as percentage and then keywords missing and last final thoughts.
    """

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text,pdf_content,input_prompt1)
        st.subheader("Response: ")
        st.write(response)
    else:
        st.write("Please Upload the pdf")
elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text,pdf_content,input_prompt2)
        st.subheader("Response: ")
        st.write(response)
    else:
        st.write("Please Upload the pdf")