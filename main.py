from Embedder import Embedder
from Model import Model
from Utilities import appendVideo
import streamlit as st
import os, time
from streamlit_chat import message
from langchain.memory import ConversationBufferMemory

def main():

    # load_dotenv()
    # API_KEY = os.environ['OPENAI_API_KEY']
    API_KEY = st.secrets['OPENAI_API_KEY']

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Tell me about what you want to listen, or how you feel and I will assist you! " +  " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey!"]

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
                                return_messages=True, output_kePy="answer", input_key="question")
        
    if 'videos' not in st.session_state:
        st.session_state['videos'] = []

    input_csv = "dataset"
 
    start_time = time.time()

    embedder = Embedder(API_KEY)
    vectorstore = embedder.getVectorStore(input_csv)
    retriever = vectorstore.as_retriever()
    model = Model(retriever)

    print("Model generation ",f"Total execution time: {time.time() - start_time} seconds")

    # CHAT INTERFACE

    #container for the chat history
    response_container = st.container()
    #container for the user's text input
    user_container = st.container()

    def conversational_chat(query):
        inputs = {'question': query}
        result = model.call({"question": inputs})
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


if __name__ == "__main__":
    main()
