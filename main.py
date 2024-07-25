import streamlit as st 
from  streamlit_option_menu import option_menu
from contextlib import contextmanager
import pandas as pd
from chat_history_model import model_ollama_buffer,model_gemma2_buffer
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
st.set_page_config(layout='wide',page_title='CHAT AI')
from RAG_ import CustomDataChatbot
from RAG_copy  import CustomDataChatbot2

sidebar = st.sidebar

sidebar.title('AI Assistants & RAG')


with sidebar:
    selected_menu = option_menu('Main Menu',['Document Reading','Assistants','Chat History'],
    icons =  ['people-fill','robot'],
    menu_icon = 'cast',default_index = 0,
    styles = {
        'container': {'padding':'0!important','background-color':'grey'},
        'icon':{'color':'orange'},
        'nav-link': {'text-align':'left','margin':'0px','--hover-color':'#eee','background-color':'grey'},
        'nav-link-selected': {'background-color':'black'}
    })

if selected_menu == 'Document Reading':
    rag_expander = st.sidebar.expander('RAG & Embedding',expanded = True)

    with rag_expander:
        rag_options = ['llama3','gemma']
        rag_choice = st.radio(
            'Select Model',
            rag_options,
            index = None, key='rag'
        )

elif selected_menu == 'Assistants':
    assistant_expander = st.sidebar.expander('Assistants',expanded =False)
    if hasattr(st.session_state,'chat_history'):
        del st.session_state['chat_history']
    with assistant_expander:
        assistant_options = ['llama3-assistant','gemma-assistant']        
        rag_choice = st.radio(
            'Select Model',
            assistant_options,
            index = None, key='rag'
        )

elif selected_menu == 'Chat History':
    if hasattr(st.session_state,'chat_history'):
        del st.session_state['chat_history']
    history_expander = st.sidebar.expander('Chat History',expanded = True)
    with history_expander:
        assistant_options = ['llama3-history','gemma-history']        
        rag_choice = st.radio(
            'Select Model',
            assistant_options,
            index = None, key='rag'
        )


if st.session_state['rag'] == 'gemma':
    obj = CustomDataChatbot()
    obj.main()
if st.session_state['rag'] == 'llama3':
 
    obj2 = CustomDataChatbot2()
    obj2.main()


elif st.session_state['rag'] == 'llama3-history':
    st.title('Chatbot with Chat History')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are an AI named okaygpt and you are working for Mercedes Benz Buses and Truck'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
        ]
    )

    model = model_ollama_buffer()
    chain = prompt_template | model

    # Chatbot function
    def chatbot():
        
        user_input = st.chat_input('Ask me something!')
        
        if user_input:
           # with st.chat_message(name='user'):
           #     st.write(user_input)
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            response = chain.invoke({'input': user_input, 'chat_history': st.session_state.chat_history})
            st.session_state.chat_history.append({'role': 'ai', 'content': response})
            st.session_state.message_count += 1
           # with st.chat_message(name='ai'):
           #     st.write(response)
            
            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message(name=chat['role']):
                    st.write(chat['content'])
            
            # Manage chat history size
            if st.session_state.message_count > 5:
                st.session_state.chat_history = st.session_state.chat_history[2:]
                st.session_state.message_count = 5
            
    chatbot()
elif st.session_state['rag'] == 'gemma-history':
    st.title('Chatbot with Chat History')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are an AI named okaygpt and you are working for Mercedes Benz Buses and Truck'),
            MessagesPlaceholder(variable_name='chat_history'),
            ('human', '{input}'),
        ]
    )

    model = model_gemma2_buffer()
    chain = prompt_template | model

    # Chatbot function
    def chatbot():
        
        user_input = st.chat_input('Ask me something!')
        
        if user_input:
           # with st.chat_message(name='user'):
           #     st.write(user_input)
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})
            response = chain.invoke({'input': user_input, 'chat_history': st.session_state.chat_history})
            st.session_state.chat_history.append({'role': 'ai', 'content': response})
            st.session_state.message_count += 1
           # with st.chat_message(name='ai'):
           #     st.write(response)
            
            # Display chat history
            for chat in st.session_state.chat_history:
                with st.chat_message(name=chat['role']):
                    st.write(chat['content'])
            
            # Manage chat history size
            if st.session_state.message_count > 5:
                st.session_state.chat_history = st.session_state.chat_history[2:]
                st.session_state.message_count = 5
            
    chatbot()