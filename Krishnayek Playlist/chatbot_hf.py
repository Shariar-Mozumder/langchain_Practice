from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint 
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user query as perfect as possible."),
        ("user","Question:{question}")
    ]
)

##streamlit framework
st.title('Welcome to Shariar chatbot, You are chatting with my AI child')
input_text=st.text_input("Ask anything you want to know")

##ollama load
# llm=Ollama(model="llama2")
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, 
    temperature=0.5, 
    token=huggingface_api_key
)
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))