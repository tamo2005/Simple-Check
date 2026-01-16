from ..utils.logging import get_logger
logger = get_logger("RAG_model_logs")
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from src.ml.text_extraction import TextExtractor

class AdvancedRAGModel:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []
            logger.info(f"RAG model initialized with embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error initializing RAG model: {str(e)}")
            raise ValueError("Failed to initialize RAG model") from e

    def add_documents(self, documents):
        try:
            embeddings = self.embedding_model.encode(documents)
            self.index.add(embeddings)
            self.documents.extend(documents)
            logger.info(f"Documents added to index: {len(documents)}")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise ValueError("Failed to add documents") from e

    def upload_and_process_documents(self, file_paths):
        try:
            extractor = TextExtractor()
            processed_texts = []

            for file_path in file_paths:
                try:
                    if file_path.endswith(".pdf"):
                        processed_texts.extend(extractor.extract_text_from_pdf(file_path))
                    elif file_path.endswith(".docx"):
                        processed_texts.extend(extractor.extract_text_from_docx(file_path))
                    elif file_path.endswith(".txt"):
                        processed_texts.append(extractor.extract_text_from_txt(file_path))
                    else:
                        logger.warning(f"Unsupported file format: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue  # Skip problematic files

            self.add_documents(processed_texts)
            logger.info(f"Processed and added {len(processed_texts)} documents")
            return processed_texts
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise ValueError("Failed to process documents") from e

    def determine_word_multiplier(self, marks):
        """
        Adjusts the word multiplier based on marks.
        """
        if marks <= 5:
            return 20  # Generate around 20 words per mark
        elif marks <= 10:
            return 15
        elif marks <= 20:
            return 10
        else:
            return 8  # For higher marks, generate fewer words per mark

    def retrieve_context(self, query, k):
        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(query_embedding, k)
            return [self.documents[i] for i in indices[0] if i < len(self.documents)]
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise ValueError("Failed to retrieve context") from e

    def generate_answer(self, question, api_key, marks):
        try:
            genai.configure(api_key=api_key)

            word_multiplier = self.determine_word_multiplier(marks)  # Determine word multiplier based on marks
            desired_word_count = word_multiplier * marks  # Calculate desired word count

            k = 3  # Fixed number of contexts to retrieve
            contexts = self.retrieve_context(question, k)
            combined_context = " ".join(contexts)

            prompt = f"""You are an expert technical assistant designed to provide detailed answers based on given context. 
            The answer should not contain bullet points or line breaks.

            Context:
            {combined_context}

            Question: {question}

            Instructions:
            - Provide a straightforward answer with approximately {desired_word_count} words.
            - Ensure the answer is clear and concise.

            Answer:"""

            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=desired_word_count * 2,  # Allow some buffer for tokenization
                ),
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"Error generating answer: {str(e)}"
