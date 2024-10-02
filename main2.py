from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

#Previous code 
# import fitz  # For reading PDFs
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# from llama_cpp import Llama
import PyPDF2
# import c_rag
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Hugging Face API token
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize the embedding model
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to extract paragraphs from a PDF using PyPDF2
def extract_paragraphs_from_pdf(pdf_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"

    # Split the text into paragraphs (using double newlines as paragraph separators)
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]

    return paragraphs

def create_embed_db(paragraphs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(paragraphs)
    db = FAISS.from_documents(docs, embeddings)
    return db

# Function to create embeddings for each paragraph (from PDF)
def create_embeddings_for_paragraphs(pdf_paths):
    contexts = []
    embeddings = []
    
    for pdf in pdf_paths:
        paragraphs = extract_paragraphs_from_pdf(pdf)
        paragraph_embeddings = embedding_model.encode(paragraphs, convert_to_numpy=True)
        contexts.extend(paragraphs)  # Store the actual paragraphs
        embeddings.append(paragraph_embeddings)  # Store the embeddings for each paragraph
    
    return np.vstack(embeddings), contexts

# List of PDF paths
pdf_paths = ["E:/Langchain Practice/Agentic RAG/pdfs/delivery_policy.pdf", 
             "E:/Langchain Practice/Agentic RAG/pdfs/exchange_and_return_policy.pdf", 
             "E:/Langchain Practice/Agentic RAG/pdfs/order_policy.pdf",
             "E:/Langchain Practice/Agentic RAG/pdfs/Privacy_policy.pdf",
             "E:/Langchain Practice/Agentic RAG/pdfs/terms_and_conditions.pdf"]

# Generate embeddings for all paragraphs
embeddings, paragraphs = create_embeddings_for_paragraphs(pdf_paths)

# Initialize FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Function to retrieve top-k paragraphs
def retrieve_top_k_paragraphs(query, k=3):
    # Encode the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Retrieve the top k paragraphs
    top_k_paragraphs = [paragraphs[i] for i in indices[0]]
    
    return top_k_paragraphs

# Function to refine top paragraphs by applying another similarity search
def refine_top_paragraphs_with_similarity(query, top_paragraphs):
    # Encode the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    
    # Encode the top paragraphs
    top_paragraph_embeddings = embedding_model.encode(top_paragraphs, convert_to_numpy=True)
    
    # Compute similarity (dot product) between the query and the top paragraphs
    similarities = np.dot(top_paragraph_embeddings, query_embedding.T).flatten()
    
    # Find the index of the paragraph with the highest similarity score
    best_index = np.argmax(similarities)
    
    # Return the most specific paragraph (highest scored)
    return top_paragraphs[best_index]

def chunk_paragraph(paragraph, chunk_size=1024):
    """Divide the paragraph into chunks of specified size."""
    return [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]

def retrieve_most_relevant_chunk(query, paragraph):
    # Divide the paragraph into 1024-character chunks
    chunks = chunk_paragraph(paragraph)

    # Encode the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Encode all chunks to get their embeddings
    chunk_embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

    # Compute similarity (dot product) between the query and all chunks
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()

    # Find the index of the chunk with the highest similarity score
    best_index = np.argmax(similarities)

    # Retrieve the most relevant chunk
    most_relevant_chunk = chunks[best_index]
    
    return most_relevant_chunk


#from git

def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # llm = OpenAI(model_name="text-davinci-003")
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.01, 
        token=huggingface_api_key
    )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about company policies 
        based on the policy documents.
        
        Answer the following question: {question}
        By searching the following policy documents: {docs}
        
        Only use the factual information from the documents to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response

if __name__ == "__main__":
    print("Welcome to the document-based chatbot! Type 'exit' to quit.")
    
    while True:
        # Take input from the user
        query = input("You: ")
        
        # If the user types 'exit', end the loop
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Retrieve the top 3 paragraphs from the FAISS search
        # top_paragraphs = retrieve_top_k_paragraphs(query, k=3)

        # Refine the top 3 paragraphs with another similarity search
        # most_specific_paragraph = refine_top_paragraphs_with_similarity(query, top_paragraphs)

        # Specify the file name
        # file_name = "output.txt"

        # # Open the file in write mode with UTF-8 encoding
        # with open(file_name, "w", encoding="utf-8") as file:
        #     file.write(most_specific_paragraph)

        # print(f"Text has been saved to {file_name}.")

        # most_relevant_chunk = retrieve_most_relevant_chunk(query, most_specific_paragraph)

        # Specify the file name
        # file_name = "output1.txt"

        # # Open the file in write mode with UTF-8 encoding
        # with open(file_name, "w", encoding="utf-8") as file:
        #     file.write(most_relevant_chunk)

        # print(f"Text has been saved to {file_name}.")

        # Ensure the context is trimmed to fit within 1024 tokens
        # if len(most_relevant_chunk) > 1024:
        #     most_relevant_chunk = most_relevant_chunk[:1024]  # Truncate to fit

        
        # print("specific chunks: ",most_relevant_chunk)
        # Generate the final answer based on the most relevant chunk
        answer = get_response_from_query(query, most_relevant_chunk)
        # answer = llama_generation(query, most_relevant_chunk)

        # Generate the final answer based on the most specific paragraph
        # answer = generate_answers_llama1(query, most_specific_paragraph)
        # answer=retrieve_most_relevant_chunk(query,answer)

        # print(f"answer before cRAG: {answer}")

        # calling C_RAG for varification
        # is_factual=False
        # for i in range(2):

        #     import time
        #     st=time.time()
        #     print("checking in c_rag..")
        #     is_factual, fact_check_result = c_rag.fact_check_answer(most_relevant_chunk, answer)
        #     et=time.time()
        #     print("Checked in c_rag. result: ",is_factual,fact_check_result)
        #     print(f"time taken in CRAG: {st-et}")
        #     if is_factual==True:
        #         break
        #     top_paragraphs = retrieve_top_k_paragraphs(query, k=3)
        #     most_specific_paragraph = refine_top_paragraphs_with_similarity(query, top_paragraphs)
        #     most_relevant_chunk = retrieve_most_relevant_chunk(query, most_specific_paragraph)
        #     if len(most_relevant_chunk) > 1024:
        #         most_relevant_chunk = most_relevant_chunk[:1024]
        #     answer = generate_answers_llama2(query, most_relevant_chunk)
        #     answer=retrieve_most_relevant_chunk(query,answer)


        # Print the answer
        print(f"Bot: {answer}")