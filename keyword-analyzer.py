import google.generativeai as genai
import re

def generate_keywords_with_weightage(question, model_answer, marks):
    # Configure Gemini API (replace 'YOUR_API_KEY' with your actual key)
    genai.configure(api_key="AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk")
    model = genai.GenerativeModel('gemini-pro')

    # Generate AI-based answer
    response = model.generate_content(question)
    ai_generated_answer = response.text.strip() if response.text else ""

    # Ask Gemini to extract important keywords in a simple list
    prompt = (f"Based on the following question and answers, extract the most relevant keywords "
              f"The weightage of the keywords will have a range from 0.1 to 100"
              f"If the marks of the question is less, then certain keywords that defines the answer will have much more weightage than other keywords"
              f"Analyze each answers and assign weightage based on the marks of the question and importance of the keyword in the answer."
              f"Question: {question}\nModel Answer: {model_answer}\nAI Answer: {ai_generated_answer}"
              f"Consider marks of the question as: {marks}"
              f"Output Format: keyword1, keyword2, keyword3")

    response = model.generate_content(prompt)
    keywords_text = response.text.strip() if response.text else ""

    # Extract keywords from the response
    keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]

    # Assign weightage dynamically
    keyword_weights = {}
    base_weight = 100 if marks > 5 else 50  # Higher weight for longer questions
    decrement = base_weight / max(len(keywords), 1)  # Distribute weightage

    for i, keyword in enumerate(keywords):
        keyword_weights[keyword] = round(base_weight - (i * decrement), 2)  # Reduce weight progressively

    return keyword_weights

# Taking inputs from the user
question = input("Enter the question: ")
model_answer = input("Enter the model answer: ")
marks = int(input("Enter the marks for the question: "))

# Generate and print keyword weightages
keywords_with_weights = generate_keywords_with_weightage(question, model_answer, marks)
print("Generated Keywords with Weightages:", keywords_with_weights)
