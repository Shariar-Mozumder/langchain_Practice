import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint  # New import
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableMap

# from getpass import getpass

# HUGGINGFACEHUB_API_TOKEN = getpass()

# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# Load environment variables


# Your Hugging Face API key (if using Hugging Face Hub)
# huggingface_api_key = "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"  # Replace with your actual Hugging Face Hub API key
huggingface_api_key = "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"  # Replace with your actual Hugging Face Hub API key


def generate_pet_name(animal_type, pet_color):
    # Create the prompt template
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names for my pet."
    )

    # Initialize the Hugging Face model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.5, 
        token=huggingface_api_key
    )

    # Create a runnable map to combine the prompt and the LLM
    
    # Prepare the inputs for the prompt
    prompt = prompt_template_name.format(animal_type=animal_type, pet_color=pet_color)

    # Generate the response from the LLM using invoke
    response = llm.invoke(prompt)

    return response 

if __name__ == "__main__":
    print(generate_pet_name("Dog", "Black"))