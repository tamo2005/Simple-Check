from ..utils.logging import get_logger
logger = get_logger("text_processing_logs")
from transformers import pipeline

class TextProcessor:
    def __init__(self):
        logger.info("Initializing TextProcessor with T5-large model.")
        try:
            self.correction_model = pipeline("text2text-generation", model="t5-large")
        except Exception as e:
            logger.error(f"Failed to initialize text correction model: {e}")
            raise

    def process_text(self, text):
        logger.info("Processing text using TextProcessor.")
        try:
            processed_text = self.correction_model(f"correct: {text}", max_length=512, truncation=True)
            logger.debug(f"Processed text: {processed_text[0]['generated_text']}")
            return processed_text[0]['generated_text']
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            raise RuntimeError(f"Text processing failed: {e}")
