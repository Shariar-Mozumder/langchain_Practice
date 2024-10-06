import PyPDF2
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
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Hugging Face API token
huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize memory to store conversation history (limit to last 6 turns)
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# List of PDF paths
pdf_paths = [
    "C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/langchain_Practice/pdfs/Delivery policy ssl_new.pdf", 
    "C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/langchain_Practice/pdfs/Employee Policy ssl_new.pdf", 
    "C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/langchain_Practice/pdfs/Order policy ssl_new.pdf",
    "C:/Users/Lenovo/ML Notebooks/ssL/langchain practice/langchain_Practice/pdfs/Return Policy ssl_new.pdf"
]

# Step 1: Extract paragraphs from the PDF
def extract_paragraphs_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    
    paragraphs = [para.strip() for para in text.split("\n\n") if para.strip()]
    return paragraphs

# Step 2: Convert paragraphs into Documents
def create_documents_from_paragraphs(paragraphs):
    documents = [Document(page_content=paragraph) for paragraph in paragraphs]
    return documents

# Step 3: Create the Vector Database using FAISS
def create_vector_db(documents):
    embeddings = HuggingFaceEmbeddings()  # You can use HuggingFaceEmbeddings() for a free option
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

# Step 4: Perform Similarity Search
def perform_similarity_search(vector_db, query, top_k=3):
    docs_with_similar_content = vector_db.similarity_search(query, k=top_k)
    return docs_with_similar_content

def get_response_from_query(query):
    """
    Get the response based on the user's query.
    """
    all_documents = []
    
    # Loop through each PDF, extract paragraphs and convert them into Document objects
    for pdf_path in pdf_paths:
        paragraphs = extract_paragraphs_from_pdf(pdf_path)
        documents = create_documents_from_paragraphs(paragraphs)
        all_documents.extend(documents)
    
    # Step 3: Create the Vector Database for all PDFs
    vector_db = create_vector_db(all_documents)
    
    # Step 4: Perform Similarity Search
    similar_docs = perform_similarity_search(vector_db, query)

    # Initialize the LLM
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.1, 
        token=huggingface_api_key
    )

    # Create the prompt template with 2 input variables
    prompt = PromptTemplate(
        input_variables=["input"],
        template="""
        You are a helpful assistant that answers questions based on conversations and provided documents.
        
        {input}
        
        Assistant:
        """
    )

    # Combine conversation history, question, and documents into one string
    conversation_history = memory.load_memory_variables({})["history"]
    relevant_docs = "\n".join([doc.page_content for doc in similar_docs])
    
    # Combine everything into a single `input` variable
    combined_input = f"Conversation History:\n{conversation_history}\n\nQuestion:\n{query}\n\nRelevant Documents:\n{relevant_docs}"

    # Generate the final response using the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"input": combined_input})
    
    # Update the memory with the new interaction
    memory.save_context({"input": combined_input}, {"response": response})

    return response

if __name__ == "__main__":
    print("Welcome to the document-based chatbot! Type 'exit' to quit.")
    
    while True:
        query = input("You: ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        answer = get_response_from_query(query)
        print(f"Bot: {answer}")

        # Display last 6 interactions from memory
        history = memory.load_memory_variables({})
        print("\n--- Last 6 Conversations ---")
        for message in history['history']:
            print(message)
