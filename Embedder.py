import os
import io
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

class Embedder:
    """
    Embedder objects allow to create embeddings of any .csv document.
    Use getVectorStore to create a vector store from a .csv document in one line. It can be then used for the retriever.
    """

    def __init__(self, OPENAI_API_KEY):
        self.API_KEY = OPENAI_API_KEY
        self.PATH = ""
        # self.PATH = "Embeddings"

        # if not os.path.exists(self.PATH):
        #     os.mkdir(self.PATH)

    @staticmethod
    def initializeData(data_path: str):
            loader = CSVLoader(file_path=data_path, encoding="utf-8", csv_args={
                    'delimiter': '\n'})
            return loader.load()


    def getVectorStore(self, data_filename: str):
        # If there is no saved vector store, generate a new one and save it before returning it
        # Else return the saved vector store
        data_path = data_filename + ".csv"
        vector_filename = f"vectors_{data_filename}"
        vectorsPath = vector_filename + ".pkl"
        
        embeddings = OpenAIEmbeddings(openai_api_key=self.API_KEY)
    
        # If wanted data is already saved in its folder
        if(os.path.isdir(vector_filename)):
            print(f"Loading saved vectorstore from {vector_filename}")
            vectorStore = FAISS.load_local(vector_filename, embeddings, allow_dangerous_deserialization=True)
            return vectorStore
        
        # Check if data_filename exists before creating a new vector store
        if(not os.path.isfile(data_path)):
            raise RuntimeError(f"Error: No such file name as {data_filename}")  
        
        with io.open(data_path) as data_file:
            data = self.initializeData(data_path)
            vectorStore = FAISS.from_documents(data, embeddings)
            vectorStore.save_local(vector_filename)
            print(f"New vectorstore saved as {vector_filename}")
            return vectorStore
        
        

        
    
