import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Coversional QA chatbot")

api_key=st.text_input("Enter your Groq API key:",type="password")

if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.3-70b-versatile")
    session_id=st.text_input("Session ID",value="default_session")
    
    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )    
    documents=[]
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        documents.extend(loader.load())
    
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()  

    contextual_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
    contextual_prompt=ChatPromptTemplate.from_messages([
        ("system",contextual_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ])
    history_retriever=create_history_aware_retriever(llm,retriever,contextual_prompt)

    system_prompt=(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}")
    ])

    qa_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_retriever,qa_chain)

    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversional_qa_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response=conversional_qa_chain.invoke({"input": user_input},
        config={
                    "configurable": {"session_id":session_id}
                })
        st.write(st.session_state.store)
        st.write("Assistant:", response['answer'])
        st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")

    
