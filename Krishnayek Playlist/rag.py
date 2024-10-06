from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint 
import bs4
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import streamlit as st

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def txtloader(path):
    loader=TextLoader(path)
    text_documents=loader.load()
    return text_documents

def pdfloader(path):
    loader=PyPDFLoader(path)
    text_documnts=loader.load()
    return text_documnts

def webloader(web_path):
    loader=WebBaseLoader(web_path,
                         bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                             class_=("post-title","post-content","post-header")
                         )))
    text_documents=loader.load()
    return text_documents

def vecorstor(docs):
    db=FAISS.from_documents(docs,HuggingFaceEmbeddings())
    return db

def retrival(query,db):
    retrival_results=db.similarity_search(query)
    return retrival_results

def generation(documents,query):
    # prompt=ChatPromptTemplate.from_messages(
    #     [
    #         ("system","You are a helpful assistant that answers questions based on provided documents:{documents}.Only use the factual information from the documents to answer the question.If you feel like you don't have enough information to answer the question, say I don't know.Your answers should be verbose and detailed.")
    #         ("user","Question:{question}")
    #     ]
    # )
    prompt = PromptTemplate(
        input_variables=["documents", "question"],
        template="""
        You are a helpful assistant that can answer questions based on provided documents.
        Answer the following question: {question}
        By searching the following documents: {documents}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.5, 
        token=huggingface_api_key
    )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    output=chain.invoke({"documents":documents,"question":query})
    return output



# if __name__=="__main__":
    ##streamlit framework
st.title('Welcome to Shariar chatbot, You are chatting with my AI child')
query=st.text_input("Ask anything you want to know")
textfilepath=""
pdfpath=""
webpath=""

text=txtloader("C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/Krishnayek Playlist/datasources/last_sermon_of_rasul.txt")
# pdftext=pdfloader("C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/Krishnayek Playlist/datasources/seerat.pdf")
# webtext=webloader("https://shariar-mozumder.github.io/index.html")
# print(text[:50])
# print(pdftext[:50])
# print(webtext[:50])
db=vecorstor(text)
retrival_docs=retrival(query,db)
if query:
    output=generation(retrival_docs,query)
    st.write(output)




