import os

import streamlit as st
#from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from chat_history_model import model_ollama_buffer,model_gemma2_buffer
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class CustomDataChatbot:
    def __init__(self):
        self.llm = model_gemma2_buffer()

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_file):
        # Load document
        file_path = self.save_file(uploaded_file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)

        # Create embeddings and store in vectordb
        embeddings = FastEmbedEmbeddings()
        vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
        )

        # Create a custom prompt
        prompt_template = """
        System : You are an AI that named  okaygpt and you are helping to Daimler Truck employees to analyzing their documents
        Use the following pieces of context to answer the customer's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        

        Context: {context}

        Question: {question}
        AI: """

        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )


        # Create the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        return qa_chain

    def main(self):
        st.title('Chatbot with PDF Upload')

        # File uploader in the sidebar
        uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None

        if uploaded_file:
            st.session_state.qa_chain = self.setup_qa_chain(uploaded_file)

        user_input = st.chat_input('Ask me something!')

        ref_list = []

        if user_input and st.session_state.qa_chain:
            st.session_state.chat_history.append(f"Human: {user_input}")

            # Format chat history
            chat_history = "\n".join(st.session_state.chat_history[-11:])  # Last 2 exchanges

            # Combine chat history with the current question
            #full_query = f"{chat_history}\nHuman: {user_input}"
            full_query = chat_history

            result = st.session_state.qa_chain.invoke({
                "query": full_query
            })
            
            response = result["result"]
            st.session_state.chat_history.append(f"AI: {response}")

            # Display references
            ref_list = []
            page_c_l = []
            for idx, doc in enumerate(result['source_documents'], 1):
                filename = os.path.basename(doc.metadata['source'])
                page_num = doc.metadata['page']
                ref_title = f":blue[Reference {idx}: {filename} - page.{page_num}]"
                ref_list.append(ref_title)
                page_c_l.append(doc.page_content)

        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(name="user" if i % 2 == 0 else "ai"):
                st.write(message.split(": ", 1)[1])
        if ref_list != []:
            for refs,content in zip(ref_list,page_c_l):
                with st.expander(refs):
                    st.caption(content)

        # Manage chat history size (optional)
        if len(st.session_state.chat_history) > 10:  # Keep last 10 messages
            st.session_state.chat_history = st.session_state.chat_history[-10:]




