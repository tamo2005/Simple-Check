import google.generativeai as genai

def generate_keywords_with_weightage(question, model_answer, marks):
    # Configure Gemini API (replace 'YOUR_API_KEY' with your actual key)
    genai.configure(api_key='AIzaSyAN6hW728nRG6ITJWMSJYQzSrkaEDeseOk')
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate AI-based answer
    response = model.generate_content(question)
    ai_generated_answer = response.text if response.text else ""
    
    # Ask Gemini to extract important keywords and assign weightage
    prompt = (f"Based on the following question and answers, extract the most relevant keywords "
              f"and assign a weightage based on importance. Consider the marks as {marks}. "
              f"Question: {question}\nModel Answer: {model_answer}\nAI Answer: {ai_generated_answer}")
    
    response = model.generate_content(prompt)
    keyword_weights = response.text if response.text else "{}"
    
    return keyword_weights

# Taking inputs from the user
question = input("Enter the question: ")
model_answer = input("Enter the model answer: ")
marks = int(input("Enter the marks for the question: "))

# Generate and print keyword weightages
keywords_with_weights = generate_keywords_with_weightage(question, model_answer, marks)
print("Generated Keywords with Weightages:")
print(keywords_with_weights)