import google.generativeai as genai
from src.utils.logging import get_logger
logger = get_logger("gemini_ocr_processor_logs")

class TextProcessor:
    def __init__(self, api_key):
        logger.info("Initializing Gemini TextProcessor with API key.")
        try:
            genai.configure(api_key="AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk")
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI: {e}")
            raise

    def process_text(self, raw_text):
        logger.info("Processing text using Gemini AI.")
        try:
            prompt = f"""You are an expert text correction assistant. 
            Carefully review the following text and correct any spelling, 
            grammatical, or OCR-related errors. Preserve the original meaning 
            and formatting as much as possible.
            Raw Text:
            {raw_text}
            Provide the corrected text:"""
            response = self.model.generate_content(
                prompt, 
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Low temperature for more accurate corrections
                    max_output_tokens=5000
                )
            )
            logger.debug(f"Generated corrected text: {response.text.strip()}")
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return f"Error processing text: {str(e)}"