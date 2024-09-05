# Chat with Websites

This is a Streamlit application that allows users to chat with websites using AI-powered conversational retrieval. The app uses LangChain, Hugging Face embeddings, and the Groq API to create a chatbot that can answer questions based on the content of a specified website.

## Features

- Web scraping and content embedding
- Conversational AI using Groq's Gemma2-9b-It model
- Context-aware retrieval for more accurate responses
- Simple and intuitive user interface

## Requirements

- Python 3.7+
- Streamlit
- LangChain
- Hugging Face Transformers
- Chroma
- python-dotenv

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/NastyRunner13/LangChain-ChatWebsite.git
   cd LangChain-ChatWebsite
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter the URL of the website you want to chat about in the "Website URL" field.

4. Once the website content is loaded and processed, you can start chatting with the AI about the website's content.

5. Type your questions or comments in the chat input field and press Enter to receive responses from the AI.

6. To clear the chat history, click the "Clear Chat History" button.

## How it Works

1. The app uses WebBaseLoader to scrape content from the provided URL.
2. The content is split into chunks using RecursiveCharacterTextSplitter.
3. Hugging Face embeddings (all-MiniLM-L6-v2) are used to create vector representations of the text chunks.
4. The embeddings are stored in a Chroma vector database.
5. When a user asks a question, the app uses a history-aware retriever to find relevant information from the vector store.
6. The retrieved information is then used by the Groq LLM to generate a response to the user's query.

## Customization

- You can change the LLM model by modifying the `ChatGroq` initialization in the code.
- Adjust the `chunk_size` and `chunk_overlap` parameters in the `RecursiveCharacterTextSplitter` to fine-tune the text splitting process.
- Modify the system prompts in the `ChatPromptTemplate` to change the AI's behavior or personality.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
