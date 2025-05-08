# """
# Utility functions for LLM integration.
# This module handles the integration with Google's Gemini models
# for document question answering.
# """
# import os
# import logging
# from typing import List, Dict, Any, Optional
# import time

# import google.generativeai as genai
# from langchain_core.prompts import PromptTemplate
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.llms import GooglePalm
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser

# # Set up logging
# logging.basicConfig(level=logging.INFO, 
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class QAModel:
#     """Class for handling language model operations for question answering."""
    
#     def __init__(self, api_key: Optional[str] = None):
#         """
#         Initialize the QA model.
        
#         Args:
#             api_key: Gemini API key (will use environment variable if None)
#         """
#         # Use provided API key or get from environment
#         self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
#         if not self.api_key:
#             error_msg = "Gemini API key not found. Please provide an API key."
#             logger.error(error_msg)
#             raise ValueError(error_msg)
        
#         # Configure Gemini API
#         genai.configure(api_key=self.api_key)
        
#         # Initialize embeddings
#         self.embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=self.api_key
#         )
        
#         # Set up the language model
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.1,
#             google_api_key=self.api_key,
#             convert_system_message_to_human=True
#         )
        
#         logger.info("QA Model initialized successfully")
    
#     def create_retrieval_qa(self, document_text: str) -> RetrievalQA:
#         """
#         Create a retrieval QA system from document text.
        
#         Args:
#             document_text: The text extracted from the document
            
#         Returns:
#             RetrievalQA: A retrieval QA system
#         """
#         try:
#             start_time = time.time()
            
#             # Split text into chunks
#             text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 separators=["\n\n", "\n", " ", ""]
#             )
#             texts = text_splitter.split_text(document_text)
#             logger.info(f"Split document into {len(texts)} chunks")
            
#             # Create vector store
#             vectorstore = FAISS.from_texts(
#                 texts, 
#                 self.embeddings
#             )
#             logger.info("Created vector store from document chunks")
            
#             # Create retriever
#             retriever = vectorstore.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={"k": 4}  # Number of documents to retrieve
#             )
            
#             # Create prompt template for Gemini
#             template = """You are a helpful assistant that answers questions based on the provided document.
            
#             Context information from the document:
#             {context}
            
#             Question: {question}
            
#             Answer the question based ONLY on the information provided in the context. 
#             If you don't know the answer or if it's not in the context, say "I don't have enough information to answer this question."
#             Provide a detailed and helpful response.
#             """
            
#             qa_prompt = PromptTemplate.from_template(template)
            
#             # Create QA chain for Gemini
#             qa_chain = (
#                 {"context": retriever, "question": lambda x: x}
#                 | qa_prompt
#                 | self.llm
#                 | StrOutputParser()
#             )
            
#             end_time = time.time()
#             logger.info(f"Created retrieval QA in {end_time - start_time:.2f} seconds")
            
#             return qa_chain
            
#         except Exception as e:
#             logger.error(f"Error creating retrieval QA: {str(e)}")
#             raise e
    
#     def answer_question(self, qa_chain, question: str) -> str:
#         """
#         Get an answer to a question using the QA chain.
        
#         Args:
#             qa_chain: The retrieval QA chain
#             question: The question to answer
            
#         Returns:
#             str: The answer to the question
#         """
#         try:
#             start_time = time.time()
            
#             # Get answer
#             answer = qa_chain.invoke(question)
            
#             end_time = time.time()
#             logger.info(f"Generated answer in {end_time - start_time:.2f} seconds")
            
#             return answer
            
#         except Exception as e:
#             logger.error(f"Error answering question: {str(e)}")
#             return f"Error generating answer: {str(e)}"

"""
Utility functions for LLM integration.
This module handles the integration with Google's Gemini models
for document question answering.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import time

import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.llms import GooglePalm
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Replace FAISS import with Chroma
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QAModel:
    """Class for handling language model operations for question answering."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the QA model.
        
        Args:
            api_key: Gemini API key (will use environment variable if None)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            error_msg = "Gemini API key not found. Please provide an API key."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Set up the language model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=self.api_key,
            convert_system_message_to_human=True
        )
        
        logger.info("QA Model initialized successfully")
    
    def create_retrieval_qa(self, document_text: str) -> RetrievalQA:
        """
        Create a retrieval QA system from document text.
        
        Args:
            document_text: The text extracted from the document
            
        Returns:
            RetrievalQA: A retrieval QA system
        """
        try:
            start_time = time.time()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_text(document_text)
            logger.info(f"Split document into {len(texts)} chunks")
            
            # Create vector store using Chroma instead of FAISS
            vectorstore = Chroma.from_texts(
                texts, 
                self.embeddings
            )
            logger.info("Created vector store from document chunks")
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Number of documents to retrieve
            )
            
            # Create prompt template for Gemini
            template = """You are a helpful assistant that answers questions based on the provided document.
            
            Context information from the document:
            {context}
            
            Question: {question}
            
            Answer the question based ONLY on the information provided in the context. 
            If you don't know the answer or if it's not in the context, say "I don't have enough information to answer this question."
            Provide a detailed and helpful response.
            """
            
            qa_prompt = PromptTemplate.from_template(template)
            
            # Create QA chain for Gemini
            qa_chain = (
                {"context": retriever, "question": lambda x: x}
                | qa_prompt
                | self.llm
                | StrOutputParser()
            )
            
            end_time = time.time()
            logger.info(f"Created retrieval QA in {end_time - start_time:.2f} seconds")
            
            return qa_chain
            
        except Exception as e:
            logger.error(f"Error creating retrieval QA: {str(e)}")
            raise e
    
    def answer_question(self, qa_chain, question: str) -> str:
        """
        Get an answer to a question using the QA chain.
        
        Args:
            qa_chain: The retrieval QA chain
            question: The question to answer
            
        Returns:
            str: The answer to the question
        """
        try:
            start_time = time.time()
            
            # Get answer
            answer = qa_chain.invoke(question)
            
            end_time = time.time()
            logger.info(f"Generated answer in {end_time - start_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"Error generating answer: {str(e)}"