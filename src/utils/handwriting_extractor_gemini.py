import os
import google.generativeai as genai
import json
import re
import base64
from io import BytesIO
from PIL import Image
from flask import jsonify, request

# Configure Gemini API Key from environment variable
GEMINI_API_KEY = "AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk"


genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_handwriting_text(image_file):
    """
    Extracts text from a handwritten document using the Gemini API.
    
    Args:
        image_file: File object representing the uploaded image.

    Returns:
        Tuple (Response JSON, HTTP Status Code)
    """
    try:
        # Read image bytes and convert to Base64
        image_bytes = image_file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode()

        # Prepare prompt for Gemini model
        extraction_prompt = """
        You are an expert in document analysis, specializing in handwritten text extraction.

        Extract the following details:
        1. QUESTION ID: 
           - Look at the left margin for an identifier (It can be a number or a string).
           - If found, return the whole id of the question.
           - If not found, return 'Unknown'.

        2. DOCUMENT CONTENT:
           - Extract the full handwritten content.
           - Preserve original structure & meaning.
           - Replace unclear words with [UNCLEAR].

        Respond strictly in JSON format:
        {
            "question_id": "extracted_id",
            "content": "full extracted text"
        }
        """

        # Send image as base64 string to Gemini
        try:
            response = model.generate_content(
                [extraction_prompt, {"mime_type": "image/png", "data": base64_image}],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type='application/json',
                    max_output_tokens=1000
                )
            )

            # Check if response is valid
            if not response or not response.text:
                return {'error': 'No response received from Gemini API', 'status': 'failure'}, 500

            # Parse the response
            try:
                result = json.loads(response.text)
                result['question_id'] = re.sub(r'[^\d]', '', str(result.get('question_id', 'Unknown')))
                result['question_id'] = result['question_id'] or 'Unknown'
                return {'status': 'success', 'data': result}, 200
            except json.JSONDecodeError as json_error:
                return {
                    'error': f'JSON Parsing Error: {str(json_error)}', 
                    'raw_response': response.text, 
                    'status': 'failure'
                }, 500

        except Exception as api_error:
            return {
                'error': f'Gemini API Error: {str(api_error)}', 
                'status': 'failure'
            }, 500

    except Exception as e:
        return {
            'error': f'Image Processing Error: {str(e)}', 
            'status': 'failure'
        }, 500