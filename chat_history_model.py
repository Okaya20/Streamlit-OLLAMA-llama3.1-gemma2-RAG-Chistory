import pandas as pd
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
import streamlit as st

ttl_value = pd.Timedelta(minutes=30).total_seconds()

@st.cache_data(ttl=ttl_value)
def model_ollama_buffer():
    model = Ollama(model = 'llama3.1',base_url ='http://localhost:11434') 
    return model

@st.cache_data(ttl=ttl_value)
def model_gemma2_buffer():
    model = Ollama(model = 'gemma2',base_url ='http://localhost:11434') 
    return model


