"""
Document-Based Question Answering Application with Google Gemini.

This application allows users to upload documents and ask questions about them,
leveraging Google's Gemini language model to provide accurate answers based on the document's content.
"""
import os
import time
import streamlit as st
from dotenv import load_dotenv
import logging

# Import custom modules
from utils.document import DocumentProcessor
from utils.model import QAModel

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize document processor
doc_processor = DocumentProcessor(upload_dir="uploaded_docs")

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'document_text' not in st.session_state:
        st.session_state.document_text = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'file_processed' not in st.session_state:
        st.session_state.file_processed = False
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'qa_model' not in st.session_state:
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            st.session_state.qa_model = QAModel(api_key=api_key)
            logger.info("QA Model initialized successfully")
        except Exception as e:
            st.session_state.qa_model = None
            logger.error(f"Failed to initialize QA Model: {str(e)}")

def process_document(uploaded_file):
    """Process the uploaded document and prepare it for question answering."""
    try:
        start_time = time.time()
        
        # Save uploaded file
        file_path = doc_processor.save_uploaded_file(uploaded_file)
        st.session_state.uploaded_file_path = file_path
        
        # Extract text from document
        document_text = doc_processor.extract_text(file_path)
        st.session_state.document_text = document_text
        
        # Create QA chain
        if st.session_state.qa_model:
            st.session_state.qa_chain = st.session_state.qa_model.create_retrieval_qa(document_text)
        
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time
        st.session_state.file_processed = True
        
        logger.info(f"Document processed successfully in {st.session_state.processing_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        st.error(f"Error processing document: {str(e)}")
        return False

def main():
    """Main function to run the Streamlit application."""
    # Set page config
    st.set_page_config(
        page_title="Document Q&A with Gemini",
        page_icon="📄",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Application title and description
    st.title("📄 Document Question Answering with Google Gemini")
    st.markdown("""
    Upload a document (PDF or text file) and ask questions about its content.
    This application uses Google's Gemini language model to provide accurate answers based on the document.
    """)
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload a PDF or text file",
            type=["pdf", "txt"],
            help="Upload a document to ask questions about."
        )
        
        if uploaded_file:
            st.info(f"File uploaded: {uploaded_file.name}")
            
            # Process button
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    success = process_document(uploaded_file)
                    if success:
                        st.success(f"Document processed in {st.session_state.processing_time:.2f} seconds!")
                    else:
                        st.error("Failed to process document.")
        
        # API key input (with warning)
        st.divider()
        st.subheader("API Configuration")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.warning("Gemini API key not found in environment variables.")
            new_api_key = st.text_input(
                "Enter your Gemini API Key:",
                type="password",
                help="Your API key will not be stored permanently."
            )
            
            if new_api_key and st.button("Save API Key"):
                os.environ["GEMINI_API_KEY"] = new_api_key
                st.session_state.qa_model = QAModel(api_key=new_api_key)
                st.success("API key saved for this session!")
        else:
            st.success("API key found in environment variables.")
    
    # Main panel for document preview and Q&A
    if st.session_state.file_processed and st.session_state.document_text:
        # Document preview
        with st.expander("Document Preview", expanded=False):
            st.text_area(
                "Extracted Text",
                st.session_state.document_text[:5000] + 
                ("..." if len(st.session_state.document_text) > 5000 else ""),
                height=300,
                disabled=True
            )
            st.caption(f"Showing first 5000 characters of {len(st.session_state.document_text)} total characters")
        
        # Q&A interface
        st.header("Ask Questions")
        st.markdown("Type your question about the document below:")
        
        # Question input
        question = st.text_input("Your question:", key="question_input")
        
        # Answer generation
        if question and st.button("Get Answer"):
            if st.session_state.qa_chain:
                with st.spinner("Generating answer..."):
                    try:
                        answer = st.session_state.qa_model.answer_question(
                            st.session_state.qa_chain, 
                            question
                        )
                        
                        # Display answer in a nicer format
                        st.markdown("### Answer:")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
            else:
                st.error("QA system not initialized. Please check your API key.")
    
    # Instructions for first-time users
    elif not st.session_state.file_processed:
        st.info("""
        👈 Start by uploading a document (PDF or text) using the sidebar.
        
        After uploading, click "Process Document" to extract the text and prepare the Q&A system.
        
        Once processed, you can ask questions about the document content!
        """)
    
    # Footer
    st.divider()
    st.caption("Document Q&A System with Google Gemini | Made with Streamlit and LangChain")

if __name__ == "__main__":
    main()