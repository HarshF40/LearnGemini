import os
from sys import exit
from urllib import response
import google.generativeai as genai
from google.generativeai.types.permission_types import Role

history = []

def main() :
    genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history = [])

    try :
        while True :    
            prompt = input(">> ")
            response = chat.send_message(prompt,stream = True)
            for chunk in response :
                print(chunk.text)

            history.append({
                "parts": [
                    {"text": prompt},
                    {"text": response.text}
                ]
            })

    except KeyboardInterrupt :
        print(chat.history)

if __name__ == "__main__" :
    main()