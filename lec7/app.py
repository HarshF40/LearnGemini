##text to sql llm application
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import sqlite3
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##function to load Google gemini model and provide sql query as response
def get_gemini_response(question,prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content([prompt,question])
    return response.text

##function to retrieve query from the sql database
def read_sql_query(sql,db):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    return rows

##definig the prompt:
prompt = """
 You are an expert in converting English questions into SQL queries.
The SQL database is named STUDENT and has the following columns: NAME, CLASS, SECTION.

Please generate only the corresponding SQL query when asked a question. Do not include any code formatting (like ```), explanations, or the word "SQL" in the output. The query should be valid SQL syntax, without any additional text.

For example:

1. If asked "How many entries of records are present?", your SQL query should be:
   SELECT COUNT(*) FROM STUDENT;

2. If asked "Tell me all the students studying in Data Science class?", your SQL query should be:
   SELECT * FROM STUDENT WHERE CLASS = 'Data Science';

Now, respond to the following question:

dont include anything else than a sql query i need only sql query

"""

##initialising streamlit app

st.set_page_config(page_title="SQL reader")
st.header("Read SQL")
question = st.text_input("Input: ",key="input")
submit = st.button("ASK")

if submit:
    response = get_gemini_response(question,prompt)
    print("Reponse: ")
    print(response) ##print in the console
    data = read_sql_query(response,"students.db")
    st.subheader("The Response is: ")
    for row in data :
        st.header(row)