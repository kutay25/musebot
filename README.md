# musebot
### Song Recommender AI with Streamlit, Langchain and Python

Uses LangChain to do "Retrieval Augmented Generation" over the MusicCaps song description dataset.

## Try it!

#### Live demo at [musebot.streamlit.app](https://musebot.streamlit.app/) 🤗

<img src="https://github.com/kutay25/musebot/assets/20889454/22699edd-640b-4787-bf9d-f2df1f6954aa" width="600" height="900">

## Running Locally 💻
Follow these steps to run musebot locally:
#### 1. Clone the repository:
`git clone https://github.com/kutay25/musebot.git`

#### 2. Change directory to project folder:
`cd path/to/project`

#### 3. Create virtual environment:
VSCode is convenient to setup virtual environments: [Environments using the create environment command](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command)

For Mac,
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
#### 4. Install dependencies inside virtual environment:
`pip install -r requirements.txt`

#### 5. Launch Streamlit locally:
`streamlit run src/main.py`

## Details

The app works by: 
1. Taking a prompt from the user
2. Combining the prompt and the chat history into a single standalone question using a chat model
3. Feeding the standalone question into the vector-database retriever, which returns a context (4 relevant songs, their links and description)
4. Combining the standalone question and context into a single prompt, which is then finally answered by a chat model.
5. Appending the message and any relevant video to the chat interface, and repeat.

- The generation of prompts, and chaining them together is done using **LangChain**. They also have a great [tutorial](https://python.langchain.com/docs/use_cases/question_answering/chat_history) for generating Q&A applications with chat history.

- As the **retriever** (objects returning relevant chunks of text from a text or .csv source), **FAISS** is used, and is available in LangChain.

- The dataset, over which the retriever retrieves relevant songs, is based on [MusicCaps Song Description Dataset](https://www.kaggle.com/datasets/googleai/musiccaps). The dataset has been processed in the following ways: A certain number of relevant songs were picked, removing unrelated videos, like guitar song lessons or low quality videos. Afterwards, irrelevant features were removed, and each entry's title was fetched from Youtube using Youtube's API. Afterwards, the dataset is read by the **Embedder** object, who calls OpenAIEmbeddings to create a vector representation of the dataset. Finally, the FAISS library creates a vector-database/vector-store, which is then saved locally - by reloading from the saved vector-store, time costs are reduced whenever the streamlit session reupdates by interactions and the embeddings API costs are highly reduced.


