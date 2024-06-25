import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import fitz

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_dressing_rooms(Rider):
    title_template = PromptTemplate(
        input_variables=["Rider"],
        template="given this text: {Rider}, I want a list of  everything asked for that listed with a quantity under dressing room and bus stock.filter the list to only include perishable items and include all alcoholic beverages in the same list. Do note include towels or glasses."
    )
    llm = OpenAI(temperature=0.7, openai_api_key=apikey,max_tokens=1000)
    title_chain = LLMChain(llm=llm, prompt=title_template)
    response = title_chain.run({"Rider": Rider})
    return response
def create_shopping_list(list):
    title_template = PromptTemplate(
        input_variables=["list"],
        template="given this list: {list}, I want you to filter out the consumables from the list and create a list for me and do not duplicate items. Sort them by into groups of beverages,fruit,food,snacks,appliances "
    )
    llm = OpenAI(temperature=0.7, openai_api_key=apikey,max_tokens=1850)
    title_chain = LLMChain(llm=llm, prompt=title_template)
    response = title_chain.run({"list": list})
    return response
def refine_shopping_list(list):
    title_template = PromptTemplate(
        input_variables=["list"],
        template="given this list: {list}, I want you to filter out certain items: cutlery, chopping board, knife/knives, tea kettle, selection of tea, wine glasses, spoons, ice bucket, corkscrew, towels. then return the list without these items"
    )
    llm = OpenAI(temperature=0.5, openai_api_key=apikey,max_tokens=2200)
    title_chain = LLMChain(llm=llm, prompt=title_template)
    response = title_chain.run({"list": list})
    return response
def main():
    st.title("Shopping List Generator")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
       response = create_shopping_list(extract_dressing_rooms(read_pdf(uploaded_file)))
       st.write(response)


if __name__ == "__main__":
    main()