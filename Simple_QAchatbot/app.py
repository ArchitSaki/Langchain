import streamlit as st
# import openai
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot With Ollama"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

prompt=ChatPromptTemplate.from_messages([
    ("system","you are a helful assistant, please answer the given question"),
    ("human","question:{question}")
])

def get_response(llm,temperature,max_tokens,question):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=max_tokens
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return chain.invoke({'question': question})

st.title("Enhanced Q&A Chatbot With GROQ")

llm=st.sidebar.selectbox("Select Open Source model",["llama-3.3-70b-versatile"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")



if user_input :
    response=get_response(llm, temperature, max_tokens,user_input)
    st.write(response)
else:
    st.write("Please provide the user input")