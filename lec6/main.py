##################################################################################################################################################
#*************************************************************************************************************************************************

# Certainly! Here's a step-by-step guide to install CUDA and FAISS with GPU support on Windows.

# ### **Step 1: Install CUDA Toolkit**

# 1. **Download CUDA Toolkit:**
#    - Visit the [CUDA Toolkit download page](https://developer.nvidia.com/cuda-downloads).
#    - Choose the appropriate operating system, version, and architecture for your machine. For most modern systems, this will be Windows, 64-bit, and you can select the latest CUDA version (e.g., CUDA 11.x).
#    - Select the installer type (e.g., **exe (local)**) and download the installer.

# 2. **Run the CUDA Installer:**
#    - Once downloaded, run the `.exe` file to start the installation.
#    - During installation, make sure to install both the **CUDA Toolkit** and **NVIDIA Drivers** if not already installed on your system.
#    - Follow the installation instructions and complete the process. 

# ### **Step 2: Add CUDA to Your PATH Environment Variable**

# 1. **Find the CUDA Installation Path:**
#    After installation, the default path for the CUDA toolkit is:
#    ```
#    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
#    ```
#    Replace `v11.x` with the version of CUDA you installed (e.g., `v11.6`).

# 2. **Add CUDA to the System PATH:**
#    - Open the **Start Menu**, type `Environment Variables`, and click **Edit the system environment variables**.
#    - In the **System Properties** window, click the **Environment Variables** button at the bottom.
#    - In the **System variables** section, scroll down and select the **Path** variable, then click **Edit**.
#    - Click **New** and add the following paths (replace `v11.x` with your installed version):
#      ```
#      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
#      C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp
#      ```
#    - Click **OK** to save the changes.

# 3. **Verify the Installation:**
#    - Open a **new command prompt** (important to open a new one to load the updated PATH).
#    - Run the following command:
#      ```bash
#      nvcc --version
#      ```
#    - This should print the version of CUDA that you installed. If it does, you've successfully installed and set up CUDA.

# ---

# ### **Step 3: Install FAISS with GPU Support**

# Since `faiss-gpu` is best installed using Conda due to its dependencies on CUDA, I'll walk you through the Conda installation method:

# 1. **Install Anaconda (if not installed already):**
#    - Download Anaconda from the official website: [Anaconda Downloads](https://www.anaconda.com/products/distribution).
#    - Follow the installation steps for your operating system. Ensure that you check the box to add Anaconda to your PATH during installation.

# 2. **Create a New Conda Environment (optional but recommended):**
#    - Open Anaconda Prompt (or Command Prompt if Conda is added to your PATH) and run:
#      ```bash
#      conda create -n faiss_env python=3.8
#      ```
#      This creates a new environment named `faiss_env` with Python 3.8.

# 3. **Activate the Conda Environment:**
#    - Activate the new environment by running:
#      ```bash
#      conda activate faiss_env
#      ```

# 4. **Install `faiss-gpu`:**
#    - With your Conda environment active, run the following command to install `faiss-gpu`:
#      ```bash
#      conda install -c pytorch faiss-gpu
#      ```
#    - This will automatically install FAISS with GPU support, along with the necessary dependencies.

# 5. **Verify the Installation:**
#    - After installation, you can verify that FAISS is installed by running a Python script or command:
#      ```python
#      import faiss
#      print(faiss.__version__)
#      ```

# ---

# ### **Step 4: Using FAISS in Your Code**

# Now that you have installed CUDA and FAISS with GPU support, you can use it in your Python scripts. Here's an example of how to load a FAISS index and perform a search:

# ```python
# import faiss
# import numpy as np

# # Create random data for testing
# d = 64                         # Dimensionality of the vectors
# nb = 1000                      # Number of database vectors
# nq = 10                        # Number of query vectors

# # Generate random database and query vectors
# xb = np.random.random((nb, d)).astype('float32')
# xq = np.random.random((nq, d)).astype('float32')

# # Initialize the FAISS index
# index = faiss.IndexFlatL2(d)   # Use L2 distance metric
# index.add(xb)                  # Add database vectors to the index

# # Perform a search
# k = 5                           # Number of nearest neighbors to retrieve
# D, I = index.search(xq, k)      # Perform search on query vectors

# print(I)  # Print the indices of the nearest neighbors
# print(D)  # Print the distances to the nearest neighbors
# ```

# ### **Optional Step: Using FAISS without GPU**

# If you don't have a compatible GPU, you can still install FAISS in CPU mode by running:

# ```bash
# pip install faiss-cpu
# ```

# ### **Conclusion**
# Now you should have CUDA and FAISS with GPU support successfully installed and set up on your Windows machine. If you face any issues during the installation, feel free to ask for further help!

#*************************************************************************************************************************************************
################################ Dependencies yet to install #####################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from dotenv import load_dotenv
from google.generativeai import embedding
load_dotenv()

import os
import streamlit as st
import google.generativeai as genai

from PyPDF2 import PdfReader ##to read the pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter ##to split large text to smaller managable chunk
from langchain_google_genai import GoogleGenerativeAIEmbeddings ##to create vector representation of text
from langchain.vectorstores import FAISS ##Facebook Ai Similiarity Search, used to store and retrive vector embeddings
from langchain_google_genai import ChatGoogleGenerativeAI ##to build chat based Ai systems
from langchain.chains.question_answering import load_qa_chain ##to build a question answering chain
from langchain.prompts import PromptTemplate ##used to create structured prompts for guiding language models to generate desired output

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

##to generate text from the pdf provided
def pdf_to_text(pdf_docs):
    text="" ##empty text variable to store the text later
    for pdf in pdf_docs: ##it will access each pdf
        pdf_file = PdfReader(pdf) ##read each pdf
        for page in pdf_file.pages: ##it will access all the pages
            text+=page.extract_text() ##extracts and stores the text from the pdf to text variable
    return text

##to divide the text data into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    ) ##creates the chunks with the overlaps
    chunks=text_splitter.split_text(text) ##here the text_splitter has the properties for the split_text() to split the text
    return chunks

##converting chunks into vectors
def get_vector_store(text_chunks):
    enbeddings = GoogleGenerativeAIEmbeddings(model="model/embedding-001") ##converts text to numerical vectors using Google's model for the conversion
    vector_store=FAISS.from_texts(text_chunks,enbeddings) ##creates a searchable database of vectors
    vector_store.save_local("faiss_index") ##saves vector to the disk (locally)

##function to setup a conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
     You are an AI assistant that has been provided a PDF document containing specific information. Your task is to answer questions based on the content of the document only. Follow these rules:
        1. Read and analyze the PDF thoroughly.
        2. Respond concisely and accurately using information only from the document.
        3. If the question is not relevant to the PDF content or cannot be answered based on it, respond with the following message: 
           "The question is out of context for the provided document. Please ask about information contained within the document."
        4. Use professional and clear language in your responses.
        Now, here's the document: \n{context}\n.
        Begin by answering the following question: \n{question}\n.
        
        Answer:
    """
    model = ChatGoogleGenerativeAI("gemini-pro",temperature=0.3) ##temperature is scale for creativity of the model, where 0 is least creative and 1 is most creative
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"]) ##this creates a prompt using the prompt template we made and add the input varaibles we provide in the prompt_template
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt) ##creates a question answering chain,stuff means the model will process the entire context and question together
    return chain

##handling the user input
def user_input(user_question):
    enbeddings = GoogleGenerativeAIEmbeddings(model = "models/enbedding-001") ##to generate enbeddings
    new_db = FAISS.load_local("faiss_index",enbeddings,allow_dangerous_deserialization=True) ##to load pre computed enbeddings
    doc = new_db.similiarity_search(user_question) ##searches the faiss indexes for the most similiar entries to the users question to then answer the question
    chain = get_conversational_chain()
    response = chain(
            {"context" : doc, "question" : user_question},
            return_only_outputs = True
        ) ##passes the relevant context (doc) and the users question to the conversational chain
    print(response)
    st.write("Reply: ",response["output_text"])

def main():
    st.set_page_config("PDF Chat")
    st.header("Chat")
    user_question = st.text_input("Ask")
    if user_question :
        user_input(user_question)

        ##sidebar
        with st.sidebar:
            st.title("Menu: ")
            pdf_docs = st.file_uploader("Upload Files...")
            if st.button("Submit") :
                with st.spinner("Processing..."):
                    raw_text = pdf_to_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.sucess("Done")

if __name__ == "__main__":
    main()