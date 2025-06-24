# MediBot Chat

MediBot Chat is an intelligent medical question-answering assistant built with retrieval-augmented generation (RAG) and local LLaMA 3 language model integration via Ollama. The app uses FAISS and SentenceTransformers to search a custom medical dataset and Streamlit for an interactive chat UI.

---

## Features

- Semantic search over medical transcripts using FAISS
- Local LLaMA 3 integration with Ollama for natural language generation
- Streamlit-based chat interface with conversation memory
- Custom prompt engineering tailored for medical advice
- Easy setup and extensible design

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/aireynexxx/MediBot.git
```
### 2. Install Requirments
```
pip install -r requirements.txt
```

### 3. Prepare embeddings

Make sure you have your medical dataset CSV file in data/medidata.csv.

Run the embedding script to create FAISS index and metadata:
```
python embedder.py
```

### 4. Start the Streamlit app
```
streamlit run app.py
```
## Usage

- Type your health-related question in the chat input.

- MediBot will retrieve relevant context and generate a medically plausible answer.

- Use the sidebar button to clear chat history if needed.

## Credits

- UI inspired by the Streamlit official tutorial: [How to Build an LLM-Powered Chatbot](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)
- Data used: [Medical Transcriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
