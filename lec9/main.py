##modified ATS, Direct PDF text extraction

import streamlit as st
import os
import google.generativeai as genai
import PyPDF2 as pdf
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text

##extracting text from pdf
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file) ##reader contains the PDF structure, metadata, text, page count 
    text = ""
    for page_index in range(len(reader.pages)): ##reader.pages --> is a list containing all the pages in the pdf, len() --> returns the number of pages in the pdf
        page = reader.pages[page_index] ##accessing the pages
        text+=str(page.extract_text()) ##extracting the text from the pages
    return text

input_prompt="""
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and
the missing keywords with high accuracy
resume:{text}
description:{job_des}

I want the response in one single string having the structure
{{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
"""

##streamlit
st.set_page_config(page_title="ATS")
st.title("Gemini ATS")
st.text("Improve Your Resume")
job_des = st.text_input("Job Description: ",key="input")
uploaded_file = st.file_uploader("Upload PDF...",type=["pdf"],help="Please upload the PDF")
submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        text=input_pdf_text(uploaded_file)
        response = get_gemini_response(input_prompt)
        st.subheader(response)