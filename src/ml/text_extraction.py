from ..utils.logging import get_logger
from ..utils.vision_handler import VisionHandler
from ..utils.image_preprocessor import ImagePreprocessor
from ..utils.gemini_ocr_processor import TextProcessor
import PyPDF2
import docx

logger = get_logger("text_extraction_logs")

class TextExtractor:
    def __init__(self):
        try:
            self.vision_handler = VisionHandler()
            self.image_preprocessor = ImagePreprocessor()
            self.text_processor = TextProcessor(api_key="AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk")
            logger.info("TextExtractor initialized")
        except Exception as e:
            logger.error(f"Error initializing TextExtractor: {str(e)}")
            raise ValueError("Failed to initialize TextExtractor") from e

    def extract_and_process_text(self, image):
        try:
            file_content = image.read()
            file_extension = image.filename.split('.')[-1].lower()

            if file_extension not in ["jpg", "jpeg", "png", "pdf"]:
                raise ValueError("Unsupported file format. Only JPG, PNG, and PDF are allowed.")

            preprocessed_image = self.image_preprocessor.preprocess_image(file_content, f".{file_extension}")
            logger.info("File preprocessed for OCR")

            result = self.vision_handler.get_text_from_image(preprocessed_image)
        
            # Ensure we return a clean dictionary structure
            return {
                "question_id": result["question_id"],
                "processed_text": result["processed_text"]
            }

        except Exception as e:
            logger.error(f"Failed to extract and process text: {str(e)}")
            raise ValueError("Failed to extract and process text") from e


    def extract_text_from_pdf(self, file_path):
        try:
            texts = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():
                        texts.append(text)
            logger.info(f"Extracted text from PDF: {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise ValueError("Failed to extract text from PDF") from e


    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            texts = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            logger.info(f"Extracted text from DOCX: {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise ValueError("Failed to extract text from DOCX") from e


    def extract_text_from_txt(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            logger.info(f"Extracted text from TXT: {file_path}")
            return [text]
        except Exception as e:
            logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
            raise ValueError("Failed to extract text from TXT") from e