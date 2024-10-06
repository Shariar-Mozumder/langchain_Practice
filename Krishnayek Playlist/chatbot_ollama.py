from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

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
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))