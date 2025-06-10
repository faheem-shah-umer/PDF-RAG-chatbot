# PDF RAG Chatbot

This project provides a Streamlit-based chatbot interface to interact with PDF documents using Retrieval-Augmented Generation (RAG) techniques. It integrates LangGraph, OpenRouter LLMs, and a local Qdrant vector store for contextualized question answering.

## Features

- Upload PDFs and convert content (text, images, tables) into embeddings
- Query PDFs using selected OpenRouter LLMs (multimodal & text)
- Web-based chat interface via Streamlit
- Max Marginal Relevance or Similarity Search
- Cosine similarity evaluation for answer context match

## Components

- `app.py`: Main Streamlit app for model selection and chat interface.
- `upload.py`: Streamlit app for uploading and embedding PDF content into Qdrant.
- `pdf2vstore_base64.py`: Extracts and filters text/tables/images from PDFs and stores them in the vector DB(Qdrant).
- `ask_chatbotLangGraph_openrouter.py`: Core chatbot logic using LangGraph and OpenRouter.
- `ask_config_openrouter.json`: Configuration for LLMs, vector store, search method, and answer instructions.
- `config.json`: Contains paths for PDFs and vector store.
- `openrouter.env`: API key for accessing OpenRouter models (not tracked in Git).

## Configuration

- `ask_config_openrouter.json`:
  - `llm_model.models`: Available LLMs with OpenRouter IDs.
  - `llm_model.selected`: Default model to start with.
  - `search_method.selected`: Search strategy for context retrieval.
  - `answer_mode`: Instruction format to guide LLM responses.
- `config.json`:
  - `data_sources.pdf.directory`: PDF input folder (used in batch mode).
  - `vector_store.path`: Path to Qdrant vector store.

## Quickstart

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenRouter API key**

   Create `.env` file or use `openrouter.env`:
   ```env
   OPENROUTER_API_KEY=your-api-key
   ```

3. **Run the PDF uploader**
   ```bash
   streamlit run upload.py
   ```

4. **Run the chatbot interface**
   ```bash
   streamlit run app.py
   ```

## Notes

- Requires Python 3.9+
- Ensure vector DB path exists (or gets created)
- Avoid re-processing same PDFs via deduplication hash logic

## License

This project is licensed under the MIT License.
