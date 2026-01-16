import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import numpy as np
import pdfplumber  # For PDFs
from docx import Document  # For DOCX files
import os

# Comment to check
class AdvancedRAGModel:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        # Initialize embedding model and FAISS index
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []

    def add_documents(self, documents):
        """
        Add documents (model answers) to the semantic search index
        
        Args:
            documents (list): List of text documents to index
        """
        # Encode documents
        embeddings = self.embedding_model.encode(documents)
        
        # Add to FAISS index
        self.index.add(embeddings)
        self.documents.extend(documents)

    def upload_and_process_documents(self, file_paths):
        """
        Process uploaded files and extract text for indexing
        
        Args:
            file_paths (list): List of file paths to process
        
        Returns:
            list: Extracted and processed text chunks
        """
        processed_texts = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                processed_texts.extend(self._extract_text_from_pdf(file_path))
            elif file_path.endswith(".docx"):
                processed_texts.extend(self._extract_text_from_docx(file_path))
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    processed_texts.append(f.read())
            else:
                print(f"Unsupported file format: {file_path}")
        
        # Add processed texts to the RAG model
        self.add_documents(processed_texts)
        return processed_texts

    def _extract_text_from_pdf(self, file_path):
        """
        Extract text from a PDF file
        
        Args:
            file_path (str): Path to the PDF file
        
        Returns:
            list: List of extracted text chunks
        """
        texts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text())
        return texts

    def _extract_text_from_docx(self, file_path):
        """
        Extract text from a DOCX file
        
        Args:
            file_path (str): Path to the DOCX file
        
        Returns:
            list: List of extracted text chunks
        """
        doc = Document(file_path)
        return [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

    def retrieve_context(self, query, k=3):
        """
        Retrieve most semantically similar documents
        
        Args:
            query (str): Input query
            k (int): Number of top documents to retrieve
        
        Returns:
            list: Most relevant document contexts
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Return top k documents
        return [self.documents[i] for i in indices[0]]

    def generate_answer(self, question, api_key):
        """
        Generate a comprehensive answer using RAG approach
        
        Args:
            question (str): Input question
            api_key (str): Gemini API key
        
        Returns:
            str: Generated answer based on retrieved context
        """
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Retrieve relevant context
        contexts = self.retrieve_context(question)
        combined_context = " ".join(contexts)
        
        # Construct enhanced prompt
        prompt = f"""You are an expert technical assistant designed to provide detailed answers based on given context. The answer should not contain bullet points and line breaks. Generate an answer for 10 marks question.

Context:
{combined_context}

Question: {question}

Instructions:
- Provide a straightforward answer without any formatting, new lines, or bold text.
- Ensure the answer is clear and concise.

Answer:"""
        
        try:
            # Use Gemini Pro for generation
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500
                )
            )
            return response.text.strip()  # Strip any leading/trailing whitespace
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Example Usage
def main():
    # Initialize RAG model
    rag_model = AdvancedRAGModel()
    
    # Upload and process documents
    file_paths = ["325_04Stacks.pdf", "326_05Queues.pdf"]
    processed_texts = rag_model.upload_and_process_documents(file_paths)
    # print(processed_texts)
    
    
    # Question answering
    question = "What is a stack and queue in computer science?"
    api_key = "AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk"
    answer = rag_model.generate_answer(question, api_key)
    print("Answer:", answer)

if __name__ == "__main__":
    main()
