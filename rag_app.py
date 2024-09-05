import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Get Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')

llm = ChatGroq(api_key=GROQ_API_KEY, model="Gemma2-9b-It")

# Set up Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

def get_vectorstore_from_url(url):

    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    document_chunks = text_splitter.split_documents(document)

    vectorstore = Chroma.from_documents(document_chunks, embeddings)

    return vectorstore

def get_context_retriever_chain(vectorstore):

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation.")
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user's questions based on the below context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):

    retriever_chain = get_context_retriever_chain(st.session_state.vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {
            "chat_history": st.session_state.chat_history,
            "input": user_input            
        }
    )

    return response['answer']

st.set_page_config(page_title="Chat with Website", layout="wide")
st.title("Chat with Websites")

st.write("Please Enter Your Website URL:")
website_url = st.text_input("Website URL")

if website_url:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?")
        ]

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = get_vectorstore_from_url(website_url)

    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("You"):
                st.write(message.content)

