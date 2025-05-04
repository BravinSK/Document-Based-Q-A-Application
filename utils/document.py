"""
Utility functions for document processing.
This module handles document loading, text extraction,
and preparation for question answering.
"""
import os
import PyPDF2
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """A class to handle document processing operations."""
    
    def __init__(self, upload_dir: str = "uploaded_docs"):
        """
        Initialize the document processor.
        
        Args:
            upload_dir: Directory to store uploaded documents
        """
        self.upload_dir = upload_dir
        
        # Create upload directory if it doesn't exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            logger.info(f"Created upload directory: {upload_dir}")
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save an uploaded file to disk.
        
        Args:
            uploaded_file: The file uploaded through Streamlit
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.upload_dir, uploaded_file.name)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"Saved uploaded file: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise e
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from uploaded document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            str: Extracted text content
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension == '.txt':
                return self._extract_text_from_txt(file_path)
            else:
                error_msg = f"Unsupported file format: {file_extension}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            raise e
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"Processing PDF with {num_pages} pages")
                
                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    def _extract_text_from_txt(self, txt_path: str) -> str:
        """
        Extract text content from a text file.
        
        Args:
            txt_path: Path to the text file
            
        Returns:
            str: Extracted text content
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            return text.strip()
        
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            logger.warning("UTF-8 decoding failed, trying with Latin-1 encoding")
            with open(txt_path, 'r', encoding='latin-1') as file:
                text = file.read()
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from text file: {str(e)}")
            raise e
    
    def chunk_text(self, text: str, chunk_size: int = 1000, 
                  chunk_overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for processing by LLM.
        
        Args:
            text: The document text to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Simple chunking by characters with overlap
        chunks = []
        
        if len(text) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            # Adjust end to avoid cutting words
            if end < len(text):
                # Try to find the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            
            # Add the chunk
            chunks.append(text[start:end].strip())
            
            # Move start position for next chunk, considering overlap
            start = end - chunk_overlap if end < len(text) else len(text)
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks