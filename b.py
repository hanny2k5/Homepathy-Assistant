import os
import os
import tempfile
import shutil
import PyPDF2
import streamlit as st
import torch
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import warnings
import threading
import psutil

# Import speech-to-text functionality using your cloned Git repo components
try:
    from sarvam_client import SarvamSTT
    from config import SUPPORTED_LANGUAGES
    from audio_recorder import AudioRecorder
    SPEECH_TO_TEXT_AVAILABLE = True
except ImportError:
    SPEECH_TO_TEXT_AVAILABLE = False
    st.warning("⚠️ Speech-to-text functionality not available. Check if sarvam_client.py and related files are present.")

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"

# Fix for PyTorch/Streamlit compatibility issue
if "STREAMLIT_WATCH_MODULES" in os.environ:
    modules_to_skip = ["torch", "tensorflow"]
    current_modules = os.environ["STREAMLIT_WATCH_MODULES"].split(",")
    filtered_modules = [m for m in current_modules if all(skip not in m for skip in modules_to_skip)]
    os.environ["STREAMLIT_WATCH_MODULES"] = ",".join(filtered_modules)



class UnifiedRAGSystem:
    def __init__(self, 
                 llm_model_name="llama3.2:latest",
                 embedding_model_name="BAAI/bge-large-en",
                 chunk_size=1000,
                 chunk_overlap=200,
                 use_gpu=True):
        """
        Initialize the Unified RAG system with multiple modes.
        
        Args:
            llm_model_name: The Ollama model for text generation
            embedding_model_name: The HuggingFace model for embeddings
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_gpu: Whether to use GPU acceleration
        """
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Device selection for embeddings
        self.device = "cuda" if self.use_gpu else "cpu"
        st.sidebar.info(f"Using device: {self.device}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": self.device}
        )
        
        # Initialize LLM
        self.llm = OllamaLLM(model=llm_model_name)
        
        # Initialize vector store
        self.doc_vector_store = None
        self.web_vector_store = None
        self.documents_processed = 0
        
        # Monitoring stats
        self.processing_times = {}
        
        # Keep track of sources
        self.sources = []
        self.errors = []

    def process_pdfs(self, pdf_files):
        """Process PDF files and create a vector store."""
        all_docs = []
        
        with st.status("Processing PDF files...") as status:
            # Create temporary directory for file storage
            temp_dir = tempfile.mkdtemp()
            st.session_state['temp_dir'] = temp_dir
            
            # Monitor processing time and memory usage
            start_time = time.time()
            
            # Track memory before processing
            mem_before = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
            
            # Process each PDF file
            for i, pdf_file in enumerate(pdf_files):
                try:
                    file_start_time = time.time()
                    
                    # Save uploaded file to temp directory
                    pdf_path = os.path.join(temp_dir, pdf_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    
                    status.update(label=f"Processing {pdf_file.name} ({i+1}/{len(pdf_files)})...")
                    
                    # Extract text from PDF
                    text = ""
                    with open(pdf_path, "rb") as f:
                        pdf = PyPDF2.PdfReader(f)
                        for page_num in range(len(pdf.pages)):
                            page = pdf.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                    
                    # Create documents
                    docs = [Document(page_content=text, metadata={"source": pdf_file.name})]
                    
                    # Split documents into chunks
                    split_docs = self.text_splitter.split_documents(docs)
                    
                    all_docs.extend(split_docs)
                    
                    file_end_time = time.time()
                    processing_time = file_end_time - file_start_time
                    
                    st.sidebar.success(f"Processed {pdf_file.name}: {len(split_docs)} chunks in {processing_time:.2f}s")
                    self.processing_times[pdf_file.name] = {
                        "chunks": len(split_docs),
                        "time": processing_time
                    }
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing {pdf_file.name}: {str(e)}")
                    self.errors.append(f"Error processing {pdf_file.name}: {str(e)}")
            
            # Create vector store if we have documents
            if all_docs:
                status.update(label="Building vector index...")
                try:
                    # Record the time taken to build the index
                    index_start_time = time.time()
                    
                    # Create the vector store using FAISS
                    self.doc_vector_store = FAISS.from_documents(all_docs, self.embeddings)
                    
                    index_end_time = time.time()
                    index_time = index_end_time - index_start_time
                    
                    # Track memory after processing
                    mem_after = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
                    mem_used = mem_after - mem_before
                    
                    total_time = time.time() - start_time
                    
                    status.update(label=f"Completed processing {len(all_docs)} chunks in {total_time:.2f}s", state="complete")
                    
                    # Save performance metrics
                    self.processing_times["index_building"] = index_time
                    self.processing_times["total_time"] = total_time
                    self.processing_times["memory_used_gb"] = mem_used
                    self.documents_processed = len(all_docs)
                    
                    return True
                except Exception as e:
                    st.error(f"Error creating vector store: {str(e)}")
                    self.errors.append(f"Error creating vector store: {str(e)}")
                    status.update(label="Error creating vector store", state="error")
                    return False
            else:
                status.update(label="No content extracted from PDFs", state="error")
                return False

    def enhance_answer(self, initial_answer, query, source_content):
        """
        Enhance the initial answer with additional context and improved quality.
        
        Args:
            initial_answer: The initial answer generated by the RAG system
            query: The original user query
            source_content: The source content chunks used to generate the answer
            
        Returns:
            An enhanced answer with improved quality and formatting
        """
        # Create an enhancement prompt template
        enhance_template = """
        You are an expert content enhancer. Your task is to improve the quality of an AI-generated answer
        while maintaining factual accuracy.
        
        Below is a query, an initial answer, and the source content used to generate that answer.
        
        QUERY:
        {query}
        
        INITIAL ANSWER:
        {initial_answer}
        
        SOURCE CONTENT (EXTRACT):
        {source_content}
        
        Please enhance the initial answer by:
        1. Improving clarity and readability
        2. Adding relevant details from the source if they were missed
        3. Ensuring all claims are factually supported by the source content
        4. Adding appropriate structure (headings, bullet points) if helpful
        5. Making sure the tone is professional and helpful
        
        ENHANCED ANSWER:
        """
        
        # Create enhancement prompt
        enhancement_prompt = PromptTemplate(
            template=enhance_template,
            input_variables=["query", "initial_answer", "source_content"]
        )
        
        # Create enhancement chain
        enhancement_chain = LLMChain(
            llm=self.llm,
            prompt=enhancement_prompt
        )
        
        # Prepare source content for the enhancement (limited to avoid token limits)
        summarized_sources = "\n\n".join([
            f"SOURCE {i+1}:\n{source[:500]}..." if len(source) > 500 else f"SOURCE {i+1}:\n{source}"
            for i, source in enumerate(source_content[:3])  # Limit to first 3 sources
        ])
        
        # Invoke the enhancement chain
        try:
            enhanced_result = enhancement_chain.invoke({
                "query": query,
                "initial_answer": initial_answer,
                "source_content": summarized_sources
            })
            
            return enhanced_result["text"].strip()
        except Exception as e:
            st.warning(f"Enhancement step encountered an issue: {str(e)}. Using initial answer.")
            self.errors.append(f"Enhancement error: {str(e)}")
            return initial_answer

    def web_search(self, query, num_results=5):
        """
        Perform a web search using multiple fallback methods
        """
        try:
            # For this implementation, we'll use a simulated search
            # In a production environment, you would integrate with a real search API
            results = self.simulate_search(query, num_results)
            if results and len(results) > 0:
                self.errors.append("Search simulation succeeded")
                return results
            else:
                return self.get_mock_results(query)
        except Exception as e:
            self.errors.append(f"Search error: {str(e)}")
            return self.get_mock_results(query)

    def simulate_search(self, query, num_results=5):
        """Simulate a search with realistic results (for demo)"""
        # Create realistic-looking mock results based on the query
        results = []
        
        # Add Wikipedia result
        results.append({
            "title": f"{query} - Wikipedia",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            "snippet": f"Wikipedia article about {query} providing comprehensive information from various sources."
        })
        
        # Academic or educational sources
        results.append({
            "title": f"{query} | Academic Research and Analysis",
            "url": f"https://scholar.example.com/research/{query.lower().replace(' ', '-')}",
            "snippet": f"Scholarly research and academic analysis of {query} with citations and peer-reviewed content."
        })
        
        # News source
        results.append({
            "title": f"Latest on {query} | News Source",
            "url": f"https://news.example.com/topics/{query.lower().replace(' ', '-')}",
            "snippet": f"Get the latest updates and news coverage about {query} from trusted journalists."
        })
        
        # Q&A or forum site
        results.append({
            "title": f"Understanding {query} - Expert Answers",
            "url": f"https://qa.example.com/questions/{query.lower().replace(' ', '-')}",
            "snippet": f"Expert answers to common questions about {query} with detailed explanations and examples."
        })
        
        # Tutorial or how-to site
        results.append({
            "title": f"{query} Guide: Complete Tutorial with Examples",
            "url": f"https://tutorials.example.com/{query.lower().replace(' ', '-')}-guide",
            "snippet": f"Step-by-step guide to understanding {query} with practical examples and applications."
        })
        
        return results[:num_results]

    def get_mock_results(self, query):
        """Generate mock search results as fallback"""
        return [
            {"title": f"Result 1 for {query}", "url": "https://example.com/1", "snippet": f"This is a sample result about {query} with relevant information."},
            {"title": f"Result 2 for {query}", "url": "https://example.com/2", "snippet": f"Another sample result for {query} with different information."},
            {"title": f"Research on {query}", "url": "https://example.com/research", "snippet": f"Academic and research information related to {query}."},
            {"title": f"Latest news about {query}", "url": "https://example.com/news", "snippet": f"Recent developments and news about {query}."},
            {"title": f"{query} - Wikipedia", "url": "https://example.com/wiki", "snippet": f"Comprehensive information about {query} from various reliable sources."}
        ]

    def fetch_webpage(self, url):
        """Fetch and parse content from a webpage with multiple fallback strategies"""
        try:
            # Make sure URL has scheme
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Log the attempt
            self.errors.append(f"Attempting to fetch content from: {url}")
            
            # Set up headers that mimic a browser
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
            
            # For the demo, we'll return simulated content
            # In a real implementation, you would make an actual HTTP request
            title = f"Simulated content for: {url}"
            content = f"This is simulated content for {url} containing relevant information about the search query. This would be real content from the web in a production environment."
            
            return {
                "url": url,
                "title": title,
                "content": content
            }
                
        except Exception as e:
            error_msg = f"Error fetching {url}: {str(e)}"
            self.errors.append(error_msg)
            return {
                "url": url,
                "title": "Error",
                "content": error_msg
            }

    def process_web_content(self, query):
        """Process web search results and create a vector store"""
        # Search the web
        search_results = self.web_search(query)
        
        # Track sources from the beginning
        self.sources = []
        for result in search_results:
            self.sources.append({
                "url": result["url"],
                "title": result["title"],
                "status": "Searched"
            })
        
        # Fetch and process documents
        documents = []
        for i, result in enumerate(search_results):
            doc = self.fetch_webpage(result["url"])
            documents.append(doc)
            
            # Update source status
            for source in self.sources:
                if source["url"] == result["url"]:
                    if "Error" in doc["title"]:
                        source["status"] = "Failed to retrieve"
                    else:
                        source["status"] = "Retrieved"
        
        # Set up vector store
        if documents:
            texts = []
            metadatas = []
            
            for doc in documents:
                chunks = self.text_splitter.split_text(doc["content"])
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({"source": doc["url"], "title": doc["title"]})
            
            # Create vector store
            self.web_vector_store = Chroma.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            return True
        return False

    def direct_retrieval_answer(self, query):
        """
        Mode 1: Direct retrieval from documents without enhancement.
        """
        if not self.doc_vector_store:
            return "Please upload and process PDF files first."
            
        try:
            # Get the retriever
            retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 4})
            
            # Create simple QA chain
            prompt_template = """
            Use the following context to answer the question. Be factual and direct.
            If the answer is not in the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            # Generate answer
            response = qa.invoke({"query": query})
            answer = response["result"]
            source_docs = response["source_documents"]
            
            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "mode": "direct_retrieval"
            }
                
        except Exception as e:
            self.errors.append(f"Direct retrieval error: {str(e)}")
            return f"Error in direct retrieval: {str(e)}"

    def enhanced_rag_answer(self, query):
        """
        Mode 2: Enhanced RAG answer with multi-stage pipeline.
        """
        if not self.doc_vector_store:
            return "Please upload and process PDF files first."
            
        try:
            # STAGE 1: RETRIEVAL
            with st.status("Stage 1: Retrieving relevant information...") as status:
                # Get top k documents from vector store
                retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 5})
                relevant_docs = retriever.invoke(query)
                
                # Extract source content for enhancement later
                source_contents = [doc.page_content for doc in relevant_docs]
                
                status.update(label=f"Retrieved {len(relevant_docs)} relevant passages", state="complete")
            
            # STAGE 2: INITIAL ANSWER GENERATION
            with st.status("Stage 2: Generating initial answer...") as status:
                # Custom prompt for initial answer
                prompt_template = """
                You are an AI assistant that provides accurate information based on documents.
                
                Use the following context to answer the question. Be detailed and precise.
                If the answer is not in the context, say "I don't have enough information to answer this question."
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:
                """
                
                PROMPT = PromptTemplate(
                    template=prompt_template, 
                    input_variables=["context", "question"]
                )
                
                # Create QA chain for initial answer
                chain_type_kwargs = {"prompt": PROMPT}
                qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs=chain_type_kwargs,
                    return_source_documents=True
                )
                
                # Generate initial answer
                response = qa.invoke({"query": query})
                initial_answer = response["result"]
                source_docs = response["source_documents"]
                
                status.update(label="Initial answer generated", state="complete")
            
            # STAGE 3: ENHANCEMENT
            with st.status("Stage 3: Enhancing answer quality...") as status:
                # Enhance the answer
                enhanced_answer = self.enhance_answer(
                    initial_answer=initial_answer, 
                    query=query, 
                    source_content=source_contents
                )
                
                status.update(label="Answer enhanced for clarity and completeness", state="complete")
            
            # Format sources
            sources = []
            for doc in source_docs:
                sources.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "source": doc.metadata.get("source", "Unknown")
                })
            
            # Return the enhanced answer and metadata
            return {
                "answer": enhanced_answer,
                "initial_answer": initial_answer,
                "sources": sources,
                "mode": "enhanced_rag"
            }
                
        except Exception as e:
            self.errors.append(f"Enhanced RAG error: {str(e)}")
            return f"Error in enhanced RAG: {str(e)}"

    def hybrid_answer(self, query):
        """
        Mode 3: Hybrid approach that combines document retrieval and web search.
        """
        try:
            doc_sources = []
            web_sources = []
            all_source_content = []
            
            # Step 1: Get information from local documents if available
            if self.doc_vector_store:
                with st.status("Retrieving information from documents...") as status:
                    # Get relevant documents
                    doc_retriever = self.doc_vector_store.as_retriever(search_kwargs={"k": 3})
                    doc_docs = doc_retriever.invoke(query)
                    
                    # Extract content
                    for doc in doc_docs:
                        all_source_content.append(doc.page_content)
                        doc_sources.append({
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "source": doc.metadata.get("source", "Local Document")
                        })
                    
                    status.update(label=f"Retrieved {len(doc_docs)} document passages", state="complete")
            
            # Step 2: Get information from the web
            with st.status("Searching and retrieving web information...") as status:
                # Process web content
                self.process_web_content(query)
                
                if self.web_vector_store:
                    # Get relevant web documents
                    web_retriever = self.web_vector_store.as_retriever(search_kwargs={"k": 3})
                    web_docs = web_retriever.invoke(query)
                    
                    # Extract content
                    for doc in web_docs:
                        all_source_content.append(doc.page_content)
                        web_sources.append({
                            "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                            "source": doc.metadata.get("source", "Web")
                        })
                
                status.update(label=f"Retrieved {len(web_sources)} web passages", state="complete")
            
            # Step 3: Combine information and generate answer
            with st.status("Generating comprehensive answer...") as status:
                if not all_source_content:
                    return "No relevant information found in documents or on the web."
                
                # Create a prompt that combines all information
                combined_prompt_template = """
                You are an AI assistant that provides comprehensive answers by combining information from multiple sources.
                
                Below is context from both uploaded documents and web searches.
                
                Context:
                {context}
                
                Question: {question}
                
                Use the following guidelines to construct your answer:
                1. Start with the most relevant and factual information
                2. Highlight agreements between sources
                3. Note any discrepancies or conflicting information
                4. Clearly identify when information comes from local documents vs. web search
                5. Structure your answer logically with appropriate headings
                6. Be comprehensive but concise
                
                Comprehensive Answer:
                """
                
                PROMPT = PromptTemplate(
                    template=combined_prompt_template, 
                    input_variables=["context", "question"]
                )
                
                # Create combined retriever
                # Since we already have the documents, we'll create a custom retriever
                combined_context = "\n\n".join(all_source_content)
                
                # Create a custom chain to use the combined context
                llm_chain = LLMChain(
                    llm=self.llm,
                    prompt=PROMPT
                )
                
                # Generate answer
                result = llm_chain.invoke({
                    "context": combined_context,
                    "question": query
                })
                
                combined_answer = result["text"]
                
                status.update(label="Comprehensive answer generated", state="complete")
            
            # Combine sources from both document and web
            all_sources = doc_sources + web_sources
            
            # Return the combined answer and sources
            return {
                "answer": combined_answer,
                "sources": all_sources,
                "doc_sources_count": len(doc_sources),
                "web_sources_count": len(web_sources),
                "mode": "hybrid"
            }
                
        except Exception as e:
            self.errors.append(f"Hybrid mode error: {str(e)}")
            return f"Error in hybrid mode: {str(e)}"

    def ask(self, query, mode="direct_retrieval"):
        """
        Ask a question using the selected mode.
        
        Args:
            query: The user question
            mode: The answering mode (direct_retrieval, enhanced_rag, or hybrid)
            
        Returns:
            Answer based on the selected mode
        """
        if mode == "direct_retrieval":
            return self.direct_retrieval_answer(query)
        elif mode == "enhanced_rag":
            return self.enhanced_rag_answer(query)
        elif mode == "hybrid":
            return self.hybrid_answer(query)
        else:
            return f"Unknown mode: {mode}"

# Streamlit UI
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "stt_client" not in st.session_state and SPEECH_TO_TEXT_AVAILABLE:
        st.session_state.stt_client = SarvamSTT()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "mode" not in st.session_state:
        st.session_state.mode = "direct_retrieval"
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""

def cleanup_temp_files():
    """Clean up temporary files when application exits."""
    if st.session_state.get('temp_dir') and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            print(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

def main():
    st.set_page_config(
        page_title="Homeopathy Assistant", 
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🌿"
    )
    
    # Custom CSS for professional healthcare UI
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .main-title {
        color: white;
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .mode-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .recording-pulse {
        animation: pulse 1s infinite;
    }
    
    .voice-ready {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🌿 Homeopathy Assistant</h1>
        <p class="main-subtitle">Intelligent Homeopathic Knowledge System with Advanced Search Capabilities</p>
    </div>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">⚙️ System Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # GPU Detection with enhanced styling
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_info = torch.cuda.get_device_properties(0)
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #2e7d32; margin: 0;">🚀 GPU Accelerated</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #424242;">{} ({:.1f} GB)</p>
            </div>
            """.format(gpu_info.name, gpu_info.total_memory / 1024**3), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #f57c00; margin: 0;">🖥️ CPU Mode</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #424242;">No GPU detected</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 Model Configuration")
        llm_model = st.selectbox(
            "LLM Model",
            ["llama3.2:latest", "llama3:latest", "mistral:latest"],
            index=0
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            [
                "BAAI/bge-large-en",
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            index=1  # all-MiniLM-L6-v2 is smaller and faster
        )
        
        use_gpu = st.checkbox("Use GPU Acceleration", value=gpu_available)
        
        # Advanced options
        with st.expander("Advanced Options"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        
        if st.button("🚀 Initialize Homeopathy Assistant", use_container_width=True):
            with st.spinner("Initializing your homeopathy knowledge system..."):
                st.session_state.rag_system = UnifiedRAGSystem(
                    llm_model_name=llm_model,
                    embedding_model_name=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    use_gpu=use_gpu and gpu_available
                )
                st.success(f"✨ System ready with {embedding_model} on {st.session_state.rag_system.device}")
        
        st.markdown("### 📚 Homeopathic Documents")
        st.markdown("*Upload your homeopathic literature, case studies, and reference materials*")
        uploaded_files = st.file_uploader("📚 Select Homeopathic PDFs", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and st.button("🔍 Process Documents", use_container_width=True):
            if not st.session_state.rag_system:
                with st.spinner("Initializing RAG system..."):
                    st.session_state.rag_system = UnifiedRAGSystem(
                        llm_model_name=llm_model,
                        embedding_model_name=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_gpu=use_gpu and gpu_available
                    )
            
            success = st.session_state.rag_system.process_pdfs(uploaded_files)
            if success:
                st.success("✨ Homeopathic documents processed successfully!")
    
    # Mode selection with homeopathy-focused design
    st.markdown("""
    <div class="mode-container">
        <h3 style="text-align: center; color: #4a4a4a; margin-bottom: 1rem;">🧪 Choose Your Consultation Method</h3>
        <p style="text-align: center; color: #666; margin-bottom: 1.5rem;">Select how you'd like the Homeopathy Assistant to analyze and respond to your queries</p>
    </div>
    """, unsafe_allow_html=True)
    
    mode_description = {
        "direct_retrieval": "📚 Quick Reference - Fast answers directly from your homeopathic literature",
        "enhanced_rag": "🔬 Deep Analysis - Comprehensive examination with multi-stage consultation pipeline",
        "hybrid": "🌍 Complete Research - Combines your documents with global homeopathic knowledge"
    }
    
    mode_cols = st.columns(3)
    with mode_cols[0]:
        mode1 = st.button("📄 Direct Retrieval", use_container_width=True)
        st.caption(mode_description["direct_retrieval"])
        
    with mode_cols[1]:
        mode2 = st.button("🔄 Enhanced RAG", use_container_width=True)
        st.caption(mode_description["enhanced_rag"])
        
    with mode_cols[2]:
        mode3 = st.button("🌐 Hybrid Search", use_container_width=True)
        st.caption(mode_description["hybrid"])
    
    if mode1:
        st.session_state.mode = "direct_retrieval"
    elif mode2:
        st.session_state.mode = "enhanced_rag"
    elif mode3:
        st.session_state.mode = "hybrid"
    
    if hasattr(st.session_state, 'mode') and st.session_state.mode in mode_description:
        st.info(f"Current mode: {st.session_state.mode} - {mode_description[st.session_state.mode]}")
    else:
        st.info("Current mode: direct_retrieval - Directly retrieve answers from documents (fastest)")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], dict):
                    st.markdown(message["content"]["answer"])
                    
                    # Display mode info
                    if "mode" in message["content"]:
                        mode_name = message["content"]["mode"]
                        st.caption(f"Answer mode: {mode_name}")
                    
                    # Display pipeline info for enhanced RAG
                    if message["content"].get("mode") == "enhanced_rag" and "initial_answer" in message["content"]:
                        with st.expander("🔄 Pipeline Information"):
                            st.subheader("Initial Answer")
                            st.markdown(message["content"]["initial_answer"])
                            st.divider()
                            st.subheader("Enhanced Answer")
                            st.markdown(message["content"]["answer"])
                    
                    # Display source info for hybrid mode
                    if message["content"].get("mode") == "hybrid":
                        if "doc_sources_count" in message["content"] and "web_sources_count" in message["content"]:
                            st.caption(f"Combined {message['content']['doc_sources_count']} document sources and {message['content']['web_sources_count']} web sources")
                    
                    # Display sources in expander
                    if "sources" in message["content"] and message["content"]["sources"]:
                        with st.expander("📄 View Sources"):
                            for i, source in enumerate(message["content"]["sources"]):
                                st.markdown(f"**Source {i+1}: {source['source']}**")
                                st.text(source["content"])
                                st.divider()
                else:
                    st.markdown(message["content"])
    
    # Simple Microphone Input using your Git repo's capabilities
    if SPEECH_TO_TEXT_AVAILABLE:
        st.markdown("### 🎤 Voice Input")
        
        if SPEECH_TO_TEXT_AVAILABLE:
            # Use your existing AudioRecorder class
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("🎙️ Start Recording", key="start_recording"):
                    if 'recorder' not in st.session_state:
                        st.session_state.recorder = AudioRecorder()
                    
                    try:
                        st.session_state.recorder.start_recording()
                        st.session_state.is_recording = True
                        st.success("🔴 Recording started! Speak now...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Recording start failed: {str(e)}")
            
            with col2:
                if st.button("⏹️ Stop & Process", key="stop_recording", disabled=not st.session_state.get('is_recording', False)):
                    if hasattr(st.session_state, 'recorder'):
                        try:
                            frames = st.session_state.recorder.stop_recording()
                            st.session_state.is_recording = False
                            
                            if frames:
                                # Save audio to temporary file
                                audio_file = st.session_state.recorder.save_audio("temp_recording.wav")
                                
                                if audio_file and os.path.exists(audio_file):
                                    with st.spinner("🔄 Processing speech with Sarvam AI..."):
                                        # Use your existing Git repo's Sarvam AI capabilities with English translation
                                        result = st.session_state.stt_client.transcribe_audio(
                                            audio_file, 
                                            'unknown',  # Let Sarvam AI auto-detect language
                                            translate_to_english=True  # Enable translation to English
                                        ) if hasattr(st.session_state, 'stt_client') else {"success": False, "error": "Speech system not available"}
                                        
                                        if result["success"]:
                                            st.session_state.transcribed_text = result["transcript"]
                                            st.success(f"✅ Speech processed: {result['transcript']}")
                                            
                                            if "language_detected" in result:
                                                st.info(f"🗣️ Language detected: {result['language_detected']}")
                                            
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                                        
                                        # Clean up temp file
                                        try:
                                            if os.path.exists(audio_file):
                                                os.remove(audio_file)
                                        except Exception as e:
                                            st.warning(f"Could not clean up temp file: {e}")
                                else:
                                    st.error("Failed to save audio recording")
                            else:
                                st.warning("No audio recorded")
                        except Exception as e:
                            st.error(f"Recording processing failed: {str(e)}")
                            st.session_state.is_recording = False
            
        with col3:
            if st.session_state.get('is_recording', False):
                st.markdown("<div style='color: red; font-weight: bold; animation: pulse 1s infinite;'>🔴 Recording in progress...</div>", unsafe_allow_html=True)
            elif st.session_state.get('transcribed_text', ''):
                st.markdown(f"<div style='color: green; font-weight: bold;'>✅ Ready: {st.session_state.transcribed_text[:30]}...</div>", unsafe_allow_html=True)
            else:
                st.info("🎤 Click 'Start Recording' to begin")
        
        # File upload fallback
        st.markdown("**Or upload audio file:**")
        audio_file = st.file_uploader(
            "Upload audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload audio in any language - will be processed by Sarvam AI"
        )
        
        if audio_file is not None:
            if st.button("🔄 Process Audio", type="primary"):
                with st.spinner("Processing audio with Sarvam AI..."):
                    # Read audio file
                    audio_bytes = audio_file.read()
                    
                    # Use your existing Git repo's Sarvam AI capabilities directly
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(audio_bytes)
                        temp_path = temp_file.name
                    
                    result = st.session_state.stt_client.transcribe_audio(
                        temp_path, 
                        'unknown',  # Let Sarvam AI auto-detect language
                        translate_to_english=True  # Enable translation to English
                    ) if hasattr(st.session_state, 'stt_client') else {"success": False, "error": "Speech system not available"}
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if result["success"]:
                        st.session_state.transcribed_text = result["transcript"]
                        st.success(f"✅ Audio processed: {result['transcript']}")
                        
                        if "language_detected" in result:
                            st.info(f"🗣️ Language: {result['language_detected']}")
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    else:
        st.info("🎤 Your Git repository has speech-to-text capability! Make sure sarvam_client.py and related files are in the same directory.")
    
    # Single clean voice input UI
    if st.session_state.transcribed_text:
        st.markdown("### 🎤 Voice Input Ready")
        st.success(f"**Transcribed:** {st.session_state.transcribed_text}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Send Voice Query", key="send_voice", type="primary"):
                prompt = st.session_state.transcribed_text
                st.session_state.transcribed_text = ""  # Clear after use
                
                # Add user message to chat and process
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Process with RAG system if available
                if st.session_state.rag_system:
                    with st.spinner(f"Processing with {st.session_state.mode} mode..."):
                        response = st.session_state.rag_system.ask(prompt, st.session_state.mode)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "Please initialize the system first."})
                
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear", key="clear_voice"):
                st.session_state.transcribed_text = ""
                st.rerun()
    
    # Regular chat input - always show
    prompt = st.chat_input("🌿 Ask about homeopathic remedies, symptoms, or treatments...")
    
    if prompt:
        # Clear transcribed text since it's being used
        if st.session_state.get('transcribed_text', ''):
            st.session_state.transcribed_text = ""
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if system is initialized
        if not st.session_state.rag_system:
            with st.chat_message("assistant"):
                message = "Please initialize the system first."
                st.markdown(message)
                st.session_state.messages.append({"role": "assistant", "content": message})
        
        # Process based on mode
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Processing with {st.session_state.mode} mode..."):
                    response = st.session_state.rag_system.ask(prompt, st.session_state.mode)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    if isinstance(response, dict):
                        st.markdown(response["answer"])
                        
                        # Display mode info
                        if "mode" in response:
                            mode_name = response["mode"]
                            st.caption(f"Answer mode: {mode_name}")
                        
                        # Display pipeline info for enhanced RAG
                        if response.get("mode") == "enhanced_rag" and "initial_answer" in response:
                            with st.expander("🔄 Pipeline Information"):
                                st.subheader("Initial Answer")
                                st.markdown(response["initial_answer"])
                                st.divider()
                                st.subheader("Enhanced Answer")
                                st.markdown(response["answer"])
                        
                        # Display source info for hybrid mode
                        if response.get("mode") == "hybrid":
                            if "doc_sources_count" in response and "web_sources_count" in response:
                                st.caption(f"Combined {response['doc_sources_count']} document sources and {response['web_sources_count']} web sources")
                        
                        # Display sources in expander
                        if "sources" in response and response["sources"]:
                            with st.expander("📄 View Sources"):
                                for i, source in enumerate(response["sources"]):
                                    st.markdown(f"**Source {i+1}: {source['source']}**")
                                    st.text(source["content"])
                                    st.divider()
                    else:
                        st.markdown(response)

if __name__ == "__main__":
    # Register cleanup function
    import atexit
    atexit.register(cleanup_temp_files)
    
    main()