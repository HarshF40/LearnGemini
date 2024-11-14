from dotenv import load_dotenv
load_dotenv()

import streamlit as stream
import google.generativeai as genai
import os

genai.configure(api_key = os.getenv("GOOGLE_GEMINI_API"))
model = genai.GenerativeModel("gemini-pro")

##this is used to store the history
chat = model.start_chat(history=[]) ##history will be stored in the history list
## we can store the chat history in the streamlit too

def get_response(prompt):
    response = chat.send_message(prompt,stream=True) ##!!here we used send message rather than doing model.generate_content(prompt)
    ##stream=True will display the content as it is generated on the go
    return response

##initializing the streamlit

stream.set_page_config(page_title="Chat History") ##to configure the page
stream.header("Chat") ##the header which will be dispayed on the main page

##checking if chat_history is in session_state
if 'chat_history' not in stream.session_state :
    stream.session_state['chat_history']= []
    ##this is because streamlit pages refreshes everytime any interaction is made... so to save it we store the chat history in the session
    ##of the streamlit

##input Box and submit button in streamlit
input = stream.text_input("Input:",key="input") ##creates a text input box
submit = stream.button("Ask") ##creates a button called submit on the page

if submit and input:
    response = get_response(input)
    stream.session_state['chat_history'].append(("You",input))##add user queries and geminis responses in the session chat history
    ##printing response
    stream.subheader("Response: ")
    for chunk in response:
        stream.write(chunk.text)
        stream.session_state['chat_history'].append(("Gemini",chunk.text))

##Displaying the chat history on the page
stream.subheader("History: ")
for role,text in stream.session_state['chat_history']: ##role,text because chat history is stored in key:value pair
    stream.write(f"{role}:{text}")