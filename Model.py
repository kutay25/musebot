import streamlit as st
from streamlit_chat import message
from langchain.schema import format_document, Document
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda


class Model:

    # Template for creating the standalone question
    std_q_template = """ 
    In your Standalone Question you must explicitly state what is the type of the question:
    Case 1) The user is currently asking for a song recommendation. If so, include this in your standalone question: "The user is asking for a song recommendation from the RAG Context part of input. The keywords that describe the song are: (Insert keywords)"
    Example situation for Case 1) might be: "The user is asking for a song recommendation from the RAG Context part of input. The keywords that describe the song are: Instrumental, calming, slow, ambient."
    Case 2) The user is currently NOT asking for a song recommendation in the "Follow Up Input". If so, write Standalone Question in this format: First, summarize the chat history, include the title of latest mentioned song. 
    Then, write that the user wants does not expect a new recommendation from the RAG Context part of your input. Write that the user expects you to answer the following question, and thene write "(here the Follow Up Input)"  
    Examples situations for Case 2) might be: "The user wants to chat about this particular topic", "The user wants to discuss the latest recommended song, (the song name)" and so on.

    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. Answer according to either Case 1) or Case 2).
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    
    # Template for answering from the standalone question + retriever context
    ans_template = """If you need to answer your question using the RAG Context, here it is below. Just choose ONE song. You must ignore this RAG Context if the user does not want a new song recommendation. 
    RAG Context: 
    {context}
    (End of RAG Context)

    Your instructions:
    1. You have previously summarized the chat history as part of "Summarized Question"
    2. Now, answer this Summarized Question in the Answer section, be chatty, but no yapping. 
    3. Never, never use the word "Context" or "RAG Context"! 
    4. If you are recommending a new song, say that you are recommending upon their preferences.

    Summarized Question: {question}
    
    (End of Summarized Question)
    
    Answer:
    """

    def call(self, prompt: str):
        return self.chain.invoke({"question": prompt})

    def __init__(self, retriever):
    
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.std_q_template)

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

        ANSWER_PROMPT = ChatPromptTemplate.from_template(self.ans_template)
        
        def _combine_documents(
            docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
        ):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        # 4 Compononets of the Chain in order
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
        
        self.chain = loaded_memory | standalone_question | retrieved_documents | answer

