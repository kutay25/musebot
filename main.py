from Embedder import Embedder
import streamlit as st
import os
import csv
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.schema import format_document, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import tempfile
import time

def main():

    load_dotenv()
    API_KEY = os.environ['OPENAI_API_KEY']

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Tell me about what you want to listen, or how you feel and I will assist you! " +  " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
                                return_messages=True, output_kePy="answer", input_key="question")

 
    start_time = time.time()
    embedder = Embedder(API_KEY)
    input_csv = "musiccaps-removed-unwanted"
    vectorstore = embedder.getVectorStore(input_csv)
    retriever = vectorstore.as_retriever()
    print("getVectorStore()",f"Total execution time: {time.time() - start_time} seconds")



    _template = """ 
    In your Standalone Question you must explicitly state what is the type of the question:
    1) The user is asking for a song recommendation. If so, include this in your standalone question: "The user is asking for a song recommendation from the RAG Context part of input."
    2) The user is currently NOT asking for a song recommendation in the "Follow Up Input". If so, write Standalone Question in this format: First, summarize the chat history, include the title of latest mentioned song. Then, write that the user wants does not expect a new recommendation from the RAG Context part of your input. Write that the user expects you to answer the following question. "(Insert here the Follow Up Input)"  
    Examples situations for 2) might be: "The user wants to chat about a particular topic", "The user wants to discuss the latest recommended song" and so on.

    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(
        docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
    ):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)


    template = """If you need to answer your question using the RAG Context, here it is below. Just choose ONE song. You must ignore this RAG Context if the user does not want a new song recommendation. 
    RAG Context: 
    {context}
    (End of RAG Context)

    You have previously summarized the chat history as part of "Question with Context."
    Now, answer this Question with Context and Answer section, be chatty, but no yapping. Never, never use the word "Context" or "RAG Context"! If you are recommending a new song, say that you are recommending upon their preferences.

    Question with Context: {question}
    
    (End of Question with Context)
    
    Answer:
    """
    ANSWER_PROMPT = ChatPromptTemplate.from_template(template)


    loaded_memory = RunnablePassthrough.assign(
        chat_history=RunnableLambda(st.session_state.memory.load_memory_variables) | itemgetter("history"),
    )
    standalone_question = {
        "standalone_question": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),
        }
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    }
    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"],
    }
    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": itemgetter("question"),
    }
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
        "docs": itemgetter("docs"),
    }
    final_chain = loaded_memory | standalone_question | retrieved_documents | answer

   
    # CHAT INTERFACE

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    user_container = st.container()

    def conversational_chat(query):
        inputs = {'question': query}
        result = final_chain.invoke({"question": inputs})
        st.session_state.memory.save_context(inputs, {"answer": result["answer"].content})
        st.session_state['history'].append((query, result["answer"]))
        
        return result["answer"].content

    with user_container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input(label="Prompt:",placeholder="Talk from here", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)


    if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
                    appendVideo(st.session_state['generated'][i], input_csv)

def appendVideo(generated_message, input_csv):
    input_csv = input_csv + ".csv"
    with open(input_csv, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        titles = [(row['title'], row['ytid']) for row in reader]

        for title in titles:   
            if title[0] in generated_message:
                st.video("https://youtube.com/watch?v="+title[1])

if __name__ == "__main__":
    main()