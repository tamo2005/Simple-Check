from google.cloud import vision
from google.cloud.vision_v1 import types
from src.utils.logging import get_logger
from src.utils.region_processor import AnswerScriptProcessor
logger = get_logger("vision_handler_logs")

class VisionHandler:
    def __init__(self, credentials_path="config/vision_key.json"):
        self.client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)
        self.answer_processor = AnswerScriptProcessor()

    def get_text_from_image(self, image_content):
        """
        Extracts text from specific regions of an answer script image.
        """
        try:
            image = types.Image(content=image_content)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise RuntimeError(f"Vision API Error: {response.error.message}")

            # Get image dimensions from the response
            width = response.full_text_annotation.pages[0].width
            height = response.full_text_annotation.pages[0].height
            
            # Extract text from specific regions
            question_id, answer_text = self.answer_processor.extract_regions(
                response.text_annotations, width, height)
            
            return {
                "question_id": question_id,
                "processed_text": answer_text
            }
        except Exception as e:
            raise RuntimeError(f"Error in VisionHandler: {e}")


# Modified text_extractor.py class method
def extract_and_process_text(self, image):
    try:
        file_content = image.read()
        file_extension = image.filename.split('.')[-1].lower()

        if file_extension not in ["jpg", "jpeg", "png", "pdf"]:
            raise ValueError("Unsupported file format. Only JPG, PNG, and PDF are allowed.")

        preprocessed_image = self.image_preprocessor.preprocess_image(file_content, f".{file_extension}")
        logger.info("File preprocessed for OCR")

        result = self.vision_handler.get_text_from_image(preprocessed_image)
        logger.info("Text extracted from file")

        # Process the answer text only
        processed_answer = self.text_processor.process_text(result["processed_text"])
        
        return {
            "question_id": result["question_id"],
            "processed_text": processed_answer
        }

    except Exception as e:
        logger.error(f"Failed to extract and process text: {str(e)}")
        raise ValueError("Failed to extract and process text") from e