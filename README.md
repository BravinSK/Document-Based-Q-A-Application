# Document-Based Q&A Application

A document-based question answering application built with Streamlit and Google's Gemini model. This application allows users to upload documents and ask questions about their content.

## Features

- PDF document upload and processing
- Document-based question answering using Google's Gemini model
- Interactive web interface using Streamlit

## Project Structure

```
doc-qa-app/
├── app.py             # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── document.py    # Document processing functions
│   └── model.py       # LLM integration functions
├── requirements.txt   # Dependencies
├── .env               # For API keys (add to .gitignore)
├── .gitignore         # For git repository
└── README.md          # Project documentation
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/BravinSK/Document-Based-Q-A-Application.git
   cd Document-Based-Q-A-Application
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate  
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Gemini API key in a `.env` file:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Requirements

See `requirements.txt` for a full list of dependencies.

## Usage

1. Start the application:

   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at `http://localhost:8501`.

3. Upload a document (PDF or text file) using the sidebar.

4. Click "Process Document" to extract the text and prepare the Q&A system.

5. Ask questions about the document content in the main panel.

## How It Works

1. **Document Processing:**

   - The application extracts text from uploaded PDF or text files.
   - The text is split into chunks for efficient processing.

2. **Question Answering:**
   - The application uses Google's Gemini model to generate embeddings for document chunks.
   - A vector database (FAISS) is created to store and retrieve document information.
   - When a question is asked, the most relevant document chunks are retrieved.
   - The Gemini language model generates an answer based on the relevant context.

## Dependencies

- streamlit: Web application framework
- PyPDF2: PDF processing
- langchain: Framework for LLM applications
- google-generativeai: Google Gemini API client
- langchain-google-genai: LangChain integration with Google Gemini
- faiss-cpu: Vector database for efficient similarity search
- python-dotenv: Environment variable management

## Configuration

The application uses the following environment variables:

- `GEMINI_API_KEY`: Your Google Gemini API key

You can set these variables in a `.env` file in the root directory.
