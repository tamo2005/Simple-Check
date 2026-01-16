import cv2
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
import json
import re
import base64

# Flask App Setup
app = Flask(__name__)

# Configure Gemini API Key
GEMINI_API_KEY = 'AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk'  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/extract-handwriting', methods=['POST'])
def extract_handwriting():
    """
    API endpoint to extract text from handwritten document without storing the file.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided', 'status': 'failure'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image', 'status': 'failure'}), 400

    try:
        # Read image bytes
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Resize image to reduce processing time (optional)
        max_width = 1024
        if image.shape[1] > max_width:
            scaling_factor = max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Convert image to Base64 efficiently
        _, buffer = cv2.imencode('.png', image)
        base64_image = base64.b64encode(buffer).decode()

        # Prepare prompt for Gemini model
        extraction_prompt = """
        You are an expert in document analysis, specializing in handwritten text extraction.

        Extract the following details:
        1. QUESTION ID: 
           - Look at the left margin for a identifier (It can be a number or a string).
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
        response = model.generate_content(
            [extraction_prompt, {"mime_type": "image/png", "data": base64_image}],
            generation_config=genai.types.GenerationConfig(
                response_mime_type='application/json',
                max_output_tokens=1000
            )
        )

        # Parse the response
        try:
            result = json.loads(response.text)
            result['question_id'] = re.sub(r'[^\d]', '', str(result.get('question_id', 'Unknown')))
            result['question_id'] = result['question_id'] or 'Unknown'
            return jsonify({'status': 'success', 'data': result}), 200
        except json.JSONDecodeError:
            return jsonify({'error': 'Parsing Error in Gemini Response', 'status': 'failure'}), 500

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failure'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

