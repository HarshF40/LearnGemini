import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os

from youtube_transcript_api import YouTubeTranscriptApi

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = """You are a youtube video summariser you will be taking the tarnscript text and summarsing the entire video and providing the important summary
  in points within 250 words"""

def extract_transcript_detail(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""
        for i in transcript_text:
            transcript+= " " + i["text"]
        return transcript
    except Exception as e:
        raise e

def generate_gemini_content(transcript_text,prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(transcript_text+prompt)
    return response.text

st.set_page_config(page_title="YTranscript")
st.title("Youtube Transcripter")
urL = st.text_input("Enter urL: ",key="input")

if urL:
    video_id = urL.split("=")[1]
    st.image(f"https://img.youtube.com/vi/{video_id}/0.jpg",use_container_width=True)
    if st.button("get") :
        text = extract_transcript_detail(urL)
        if text :
            response = generate_gemini_content(text,prompt)
            st.markdown("Response")
            st.write(response)